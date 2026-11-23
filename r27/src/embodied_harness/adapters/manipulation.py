"""Measured handle grasp and bounded linear pull; no joint/object-state access."""
import math
from .skills import grasp_point, validate_grasp
from .gripper_state import possible_hold_after, released


PULL_SCHEMA={"description":"Grasp a freshly marked VISIBLE drawer handle from its front, then pull horizontally outward. For prismatic drawers, not lifting bowls and not rotating doors. Uses local measured RGB-D surface normal if outward_world is omitted. Select the requested handle, not the cabinet panel or drawer above it. Approaches from outside, closes jaws vertically around a horizontal handle, checks bilateral contact and pulls at constant height. Actual opening must be verified visually. Simulation only.",
 "parameters":{"type":"object","additionalProperties":False,"required":["point_id"],"properties":{
     "point_id":{"type":"string","minLength":1},
     "outward_world":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},
     "distance_m":{"type":"number","minimum":.05,"maximum":.25},
     "contact_depth_m":{"type":"number","minimum":0,"maximum":.04}}}}


def pull_direction(adapter,args):
    point=adapter.points[args['point_id']]
    vector=args.get('outward_world') or point.get('measurement',{}).get('horizontal_bar_outward_world') or point.get('measurement',{}).get('camera_facing_surface_normal_world')
    if vector is None:raise ValueError('Surface normal uncertain; inspect handle or specify observed outward_world')
    x,y,_=map(float,vector);norm=math.hypot(x,y)
    if not math.isfinite(norm) or norm<.4:raise ValueError('Need a front-facing handle surface; selected surface points upward')
    return [x/norm,y/norm,0.]


def validate_pull(adapter,args):
    validate_grasp(adapter,{'point_id':args['point_id']})
    prominence=adapter.points[args['point_id']].get('measurement',{}).get('local_depth_prominence_m')
    if prominence is not None and prominence<.005:
        raise ValueError('Selected surface has insufficient visible protrusion for this bar-grasp skill; inspect a different handle surface instead of a flush panel')
    pull_direction(adapter,args)


async def pull_handle(adapter,args,cancelled,publish):
    outward=pull_direction(adapter,args)
    depth=args.get('contact_depth_m',.015)
    point=adapter.points[args['point_id']]
    robot=adapter.last['robot'];current=robot['grip_xyz_m'];height=max(current[2],point['xyz_m'][2]+.12)
    orientation={'approach_world':[-x for x in outward],'jaw_world':[0,0,1]}
    preamble=[('open_before_route','gripper_control',{'action':'open'})]
    if height>current[2]+.006:preamble.append(('vertical_clearance','move_to',{'xyz_m':[current[0],current[1],height]}))
    preamble.append(('orient_at_clearance','move_to',{'xyz_m':[current[0],current[1],height],**orientation}))
    preamble.append(('route_outside_handle','move_to',{'xyz_m':[point['xyz_m'][0]+outward[0]*.10,point['xyz_m'][1]+outward[1]*.10,height],**orientation}))
    trace=[]
    for phase,name,parameters in preamble:
        if cancelled():return {'skill':'pull_handle','macro_status':'cancelled_before_'+phase,'state_unknown':False,'stop_confirmed':True,'feedback':{'substeps':trace}}
        raw=await adapter._thread(adapter.host.call,name,parameters);obs=adapter._unpack(raw);publish(obs)
        body=adapter.last;feedback=body.get('feedback') or {};error=body.get('error') or {};motion=feedback.get('motion') or {}
        trace.append({'phase':phase,'tool':name,'args':parameters,'feedback':feedback,'error':error,'observation_id':obs.id})
        if name=='gripper_control':adapter.possible_hold=possible_hold_after(adapter.possible_hold,'open',body)
        unknown=error.get('code')=='host_execution_error' or motion.get('execution_state') in {'unknown','state_unknown'}
        if error or motion.get('execution_state') not in {'reached','completed'} or (name=='gripper_control' and not released(body)):
            return {'skill':'pull_handle','macro_status':phase+'_failed','state_unknown':unknown,'feedback':{'substeps':trace}}
    result=await grasp_point(adapter,{'point_id':args['point_id'],'lift_m':0.,
        'approach_world':[-x for x in outward],'jaw_world':[0,0,1],
        'contact_position_tolerance_m':.04,
        'contact_offset_m':[-x*depth for x in outward],'clearance_m':.10},cancelled,publish)
    result['skill']='pull_handle'
    result['feedback']['substeps']=trace+result['feedback']['substeps']
    if result['macro_status']!='contact_established_no_lift':return result
    if cancelled():result.update(macro_status='cancelled_before_pull',stop_confirmed=True);return result
    params={'delta_m':[x*args.get('distance_m',.18) for x in outward]}
    raw=await adapter._thread(adapter.host.call,'move_to',params)
    obs=adapter._unpack(raw);publish(obs)
    feedback=adapter.last.get('feedback') or {};error=adapter.last.get('error')
    motion=feedback.get('motion') or {}
    unknown=(error or {}).get('code')=='host_execution_error' or motion.get('execution_state') in {'unknown','state_unknown'}
    result['feedback']['substeps'].append({'tool':'move_to','phase':'pull','args':params,'feedback':feedback,'error':error,'observation_id':obs.id})
    status='pull_completed_verify_visually' if not error and motion.get('execution_state')=='reached' else 'pull_partial_inspect'
    result.update(macro_status=status,state_unknown=unknown,observation_id=obs.id,stop_confirmed=cancelled() and not unknown)
    result['feedback']['skill_status']=status
    return result
