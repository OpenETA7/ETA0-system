"""Independent visual identity check, deliberately isolated from actor rationale."""
from collections import deque

IDENTITY = '''You independently compare the prior verified appearance reference with the CURRENT
selected object crop. You have no action history and no actor's identity claims: judge only these images.
Compare color, shape, height-to-width ratio, packaging and actually visible text. A bottle is not a can.
Unreadable text is not evidence. The reference is prior experience, not the current scene or location.
Return one JSON object: {"reference_appearance":"...","selected_appearance":"...",
"verdict":"match|mismatch|uncertain","reason":"..."}. Do not propose or execute an action.
A match is an appearance judgment, not task success or physical safety certification.'''


class IdentityChecker:
    def __init__(self,model,context,coordinator,store):
        self.model,self.context,self.coordinator,self.store=model,context,coordinator,store
        self.model.history=deque(maxlen=0)
        self.model.input_instruction='The image is the current selected object, enlarged. Compare it against the prior reference independently.\n'
        self.cache={}

    async def check(self,point_id):
        control_version = self.context.control_version
        adapter=self.coordinator.adapter;point=adapter.points[point_id]
        key=(point['source_packet_id'],point['camera_frame_id'],tuple(round(float(x),4) for x in point['xyz_m']))
        if key in self.cache:
            result=self.cache[key];self.store.emit('identity.cached',{'point_id':point_id,**result})
        else:
            x,y=point['pixel_xy'];bbox=(point.get('visual_candidate') or {}).get('bbox_xyxy')
            crop=[max(0,x-35),max(0,y-35),min(512,x+36),min(512,y+36)]
            if bbox:crop=[max(0,bbox[0]-8),max(0,bbox[1]-8),min(512,bbox[2]+8),min(512,bbox[3]+8)]
            async with self.coordinator.lock:
                obs=await adapter.inspect('observe',{'camera':point['camera_frame_id'],'crop_xyxy':crop},self.context.action_seq)
                self.context.ingest(obs)
            views=obs.state.get('views') or []
            if (obs.state.get('error') or not views or views[-1].get('camera_frame_id')!='zoom_preview'
                or views[-1].get('source_camera')!=point['camera_frame_id']
                or views[-1].get('original_crop_xyxy')!=crop
                or views[-1].get('source_packet_id')!=point['source_packet_id']):
                raise ValueError('Identity check requires an actual current selected-object crop')
            judge_context = {'mission':self.context.milestone.goal}
            if getattr(self.model, 'control_context', None) is not None:
                judge_context['control_version'] = control_version
            result=await self.model.decide(IDENTITY,judge_context,[obs.images[-1]])
            if result.get('verdict') not in {'match','mismatch','uncertain'}:raise ValueError('Identity check returned invalid verdict')
            self.cache[key]=result
            self.store.emit('identity.checked',{'point_id':point_id,'source_packet_id':point['source_packet_id'],**result})
        self.context.local_state['identity_check']={'point_id':point_id,**result,
            'readiness':'Measured point matched the reference; grasp_point can now use this exact point_id if picking is intended' if result['verdict']=='match' else 'Rejected identity: choose a different current region',
            'provenance':'independent_model_appearance_judgment_not_world_truth'}
        # Frame-scoped evidence survives a short-dialogue reset, but must never
        # treat a candidate number as a persistent object identity after motion.
        visual = point.get('visual_candidate') or {}
        candidate_id = visual.get('candidate_id')
        scope = self.context.latest.state.get('candidate_scope')
        if candidate_id and scope:
            ledger = self.context.local_state.get('identity_candidates') or {}
            if ledger.get('visual_scope') != scope:
                ledger = {'visual_scope': scope, 'checks': {},
                          'provenance': 'independent_model_judgments_not_ground_truth'}
            ledger['checks'][candidate_id] = {
                'verdict': result['verdict'],
                'selected_appearance': str(result.get('selected_appearance', ''))[:400],
                'reason': str(result.get('reason', ''))[:400],
            }
            ledger['checks'] = dict(list(ledger['checks'].items())[-24:])
            self.context.local_state['identity_candidates'] = ledger
        if result['verdict']!='match':
            self.context.local_state['readonly_streak']+=1
            self.context.remember('identity.rejected',{'point_id':point_id,**result,
                'next_step':'Select a different candidate or obtain clearer visual evidence; do not repeat the same rejected object claim'})
            return False
        return True


TARGET_IDENTITY = '''Independently judge whether the selected CURRENT region is an object that the
user's original mission asks the robot to acquire. You receive the original mission, a labeled
historical task-start reference, and exactly two current images: full scene then selected crop.
Use the original scene to resolve spatial descriptions; never infer current pose from that old image.
Use visible shape, packaging and readable label text. Distinguish movable mission targets from
robot parts, fixtures and background surfaces; proximity alone does not establish identity.
Do not trust an object's name merely because it is the selected region. Unreadable text is unknown.
You have no actor rationale, planner identity claim, action history or execution tools.
Return ONLY {"verdict":"match|mismatch|uncertain","selected_appearance":"visible evidence",
"reason":"why this actual region does or does not match the mission target"}.
If evidence is insufficient, use uncertain. Do not choose a replacement region or prescribe actions.
A match is only a fallible semantic judgment; it does not certify grasp safety or task success.'''


class MissionTargetJudge:
    """Memory-independent pre-grasp semantic veto with strict current-scene fences."""
    pregrasp_only = True

    def __init__(self, model, context, coordinator, store):
        self.model, self.context, self.coordinator, self.store = model, context, coordinator, store
        self.model.history = deque(maxlen=0)
        self.model.input_instruction = "Independent mission-to-selected-region check. Current images: full scene, exact selected crop.\n"
        self.cache = {}

    def stamp(self):
        from embodied_harness.runtime.coordinator import GateError
        c, m, o = self.coordinator, self.context.milestone, self.context.latest
        if (not m or not o or c.paused or c.unknown or c.running
                or c.contract_review_pending):
            raise GateError("target_judge_requires_current_idle_authority")
        if o.action_seq != self.context.action_seq or o.generation != c.adapter.generation:
            raise GateError("target_judge_observation_not_current")
        return (self.context.control_version, m.id, m.revision, m.generation, m.mission,
                self.context.action_seq, c.adapter.generation, c.pause_epoch,
                c.contract_review_epoch, o.state.get('candidate_scope'))

    def require_stamp(self, stamp, observation_id=None):
        from embodied_harness.runtime.coordinator import GateError
        if self.stamp() != stamp or (observation_id is not None and self.context.latest.id != observation_id):
            raise GateError("target_judge_evidence_or_authority_changed")

    async def check(self, point_id):
        import copy
        import hashlib
        from pathlib import Path
        from embodied_harness.runtime.coordinator import GateError
        self.context.local_state.pop('target_identity_check', None)
        reference = getattr(self.model, 'mission_reference_metadata', None)
        if not reference or not reference.get('image_sha256'):
            raise GateError("target_judge_requires_task_start_reference")
        stamp = self.stamp()
        adapter = self.coordinator.adapter
        point = copy.deepcopy(adapter.points[point_id])
        visual = point.get('visual_candidate') or {}
        candidate_id = visual.get('candidate_id')
        x, y = point['pixel_xy']
        bbox = visual.get('bbox_xyxy')
        crop = ([max(0,bbox[0]-8),max(0,bbox[1]-8),min(512,bbox[2]+8),min(512,bbox[3]+8)]
                if bbox else [max(0,x-35),max(0,y-35),min(512,x+36),min(512,y+36)])
        # A candidate number alone is never stable identity. Include immutable
        # physical/frame/contract scope; point changes within one crop can reuse
        # a semantic verdict, but every action or control change expires it.
        key = (stamp, reference['image_sha256'], point['source_packet_id'],
               point['camera_frame_id'], candidate_id, tuple(crop))
        if key in self.cache:
            self.require_stamp(stamp)
            result, evidence = self.cache[key]
            result, evidence = copy.deepcopy(result), copy.deepcopy(evidence)
            self.store.emit('target_identity.cached', {'point_id':point_id, **result, 'evidence':evidence})
        else:
            scene_obs = None
            async with self.coordinator.lock:
                self.require_stamp(stamp)
                if point['camera_frame_id'] != 'agentview':
                    scene_obs = await adapter.inspect('observe', {'camera':'agentview'}, self.context.action_seq)
                    self.require_stamp(stamp)
                    self.context.ingest(scene_obs)
                    self.require_stamp(stamp)
                obs = await adapter.inspect('observe', {'camera':point['camera_frame_id'], 'crop_xyxy':crop}, self.context.action_seq)
                self.require_stamp(stamp)
                self.context.ingest(obs)
            self.require_stamp(stamp)
            scene_obs = scene_obs or obs
            scene_views = scene_obs.state.get('views') or []
            views = obs.state.get('views') or []
            full = next((i for i,v in enumerate(scene_views) if v.get('camera_frame_id') == 'agentview'), None)
            zoom = next((i for i,v in enumerate(views) if v.get('camera_frame_id') == 'zoom_preview'
                         and v.get('source_camera') == point['camera_frame_id']
                         and v.get('original_crop_xyxy') == crop
                         and v.get('source_packet_id') == point['source_packet_id']), None)
            if (obs.state.get('error') or scene_obs.state.get('error') or full is None or zoom is None
                    or full >= len(scene_obs.images) or zoom >= len(obs.images)
                    or scene_views[full].get('source_packet_id') != point['source_packet_id']
                    or scene_views[full].get('is_current_observation') is not True):
                raise GateError('target_judge_requires_exact_current_scene_and_crop')
            paths = [scene_obs.images[full], obs.images[zoom]]
            evidence = {'observation_id':obs.id, 'scene_observation_id':scene_obs.id,
                        'source_packet_id':point['source_packet_id'],
                        'camera_frame_id':point['camera_frame_id'], 'candidate_id':candidate_id,
                        'crop_xyxy':crop, 'action_seq':self.context.action_seq,
                        'control_version':self.context.control_version,
                        'task_start_image_sha256':reference['image_sha256'],
                        'current_image_sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]}
            query = {'mission':self.context.milestone.mission, 'control_version':self.context.control_version,
                     'evidence':evidence, 'views':[scene_views[full],views[zoom]]}
            self.store.emit('target_identity.started', {'point_id':point_id,'evidence':evidence})
            result = await self.model.decide(TARGET_IDENTITY, query, paths)
            self.require_stamp(stamp, obs.id)
            if (not isinstance(result,dict) or set(result) != {'verdict','selected_appearance','reason'}
                    or result['verdict'] not in {'match','mismatch','uncertain'}
                    or any(not isinstance(result[k],str) or not 1 <= len(result[k]) <= 1600
                           for k in ('selected_appearance','reason'))):
                raise ValueError('target judge requires a bounded semantic verdict, appearance and reason')
            if result['verdict'] != 'uncertain':
                self.cache[key] = (copy.deepcopy(result),copy.deepcopy(evidence))
                if len(self.cache) > 64:
                    self.cache.pop(next(iter(self.cache)))
            self.store.emit('target_identity.checked', {'point_id':point_id, **result, 'evidence':evidence})
        self.require_stamp(stamp)
        self.context.local_state['target_identity_check'] = {
            'point_id':point_id, **result, 'evidence':evidence,
            'provenance':'independent_model_target_judgment_not_world_truth',
            'readiness':('Semantic veto cleared only; physical and contract gates still apply.'
                         if result['verdict']=='match' else 'Target not established. Reinspect the mission and choose a current region using visible evidence.')}
        scope = self.context.latest.state.get('candidate_scope')
        if candidate_id and scope:
            ledger = self.context.local_state.get('identity_candidates') or {}
            if ledger.get('visual_scope') != scope:
                ledger = {'visual_scope':scope,'checks':{},'provenance':'independent_model_judgments_not_ground_truth'}
            ledger.update(judgment_role='target_judge', control_version=self.context.control_version,
                          action_seq=self.context.action_seq)
            ledger['checks'][candidate_id] = {k:result[k] for k in ('verdict','selected_appearance','reason')}
            ledger['checks'] = dict(list(ledger['checks'].items())[-24:])
            self.context.local_state['identity_candidates'] = ledger
        if result['verdict'] != 'match':
            self.context.local_state['readonly_streak'] += 1
            self.context.remember('target_identity.rejected', {'point_id':point_id,**result,
                'next_step':'Choose or inspect another current region; do not reinterpret this veto as target identity.'})
            return False
        return True
