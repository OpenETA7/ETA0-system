"""Independent current-point destination semantics; no motion parameters or authority."""
import copy
import hashlib
from pathlib import Path

from .grounding import MissionTargetJudge
from embodied_harness.runtime.coordinator import GateError


DESTINATION = '''Independently judge whether the CURRENT marked placement point belongs to the
destination required by the user's original mission. You see a labeled historical task-start image,
then the current full scene and exact selected-region crop. The full-scene red marker and supplied
pixel_xy locate the actual anchor; a large crop containing a destination does not prove the marked
point belongs to it. Camera pixels use origin TOP LEFT. Use the original scene only for the original
spatial relation; it is not current pose evidence.
Distinguish the requested support/container from robot structure, nearby distractors, the held source
object, and unrelated supports. A region ID is only an unlabeled image annotation. Confirm the
requested semantic destination and relevant spatial relation using CURRENT visible evidence.
For placement_anchor=true, the tool estimates an opening/footprint anchor, not a measured floor;
judge whether that opening belongs to the requested destination, without claiming the 3D floor.
You receive no actor rationale, planner identity guesses, action history, or execution tools.
Return ONLY {"verdict":"match|mismatch|uncertain","selected_appearance":"visible evidence",
"reason":"why the marked anchor does or does not belong to the requested destination"}.
Use uncertain if the relation or actual anchor cannot be established. Do not choose a replacement
point, prescribe a tool or release height, or assert that an object is retained. A match is a fallible
semantic judgment only: it certifies neither collision clearance, placement stability nor task success.'''


class DestinationJudge(MissionTargetJudge):
    """Separate paid role/cache; shares the established strict authority stamp."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.input_instruction = 'Independent placement-destination check. Current images: full scene, exact anchor crop.\n'

    async def check(self, point_id):
        self.context.local_state.pop('destination_check', None)
        reference = getattr(self.model, 'mission_reference_metadata', None)
        if not reference or not reference.get('image_sha256'):
            raise GateError('destination_judge_requires_task_start_reference')
        stamp = self.stamp()
        adapter = self.coordinator.adapter
        point = copy.deepcopy(adapter.points[point_id])
        if point.get('camera_frame_id') != 'agentview':
            raise GateError('destination_judge_requires_agentview_anchor; remeasure destination in current agentview')
        visual = point.get('visual_candidate') or {}
        candidate_id = visual.get('candidate_id')
        x, y = point['pixel_xy']
        bbox = visual.get('bbox_xyxy')
        crop = ([max(0,bbox[0]-8),max(0,bbox[1]-8),min(512,bbox[2]+8),min(512,bbox[3]+8)]
                if bbox else [max(0,x-35),max(0,y-35),min(512,x+36),min(512,y+36)])
        # Unlike region appearance, anchor semantics cannot be transferred to a
        # different point within a mixed region. Physical actions expire it too.
        key = (stamp, reference['image_sha256'], point['source_packet_id'],
               point['camera_frame_id'], candidate_id, tuple(crop), tuple(point['pixel_xy']),
               tuple(point['xyz_m']), bool(point.get('placement_anchor')))
        if key in self.cache:
            self.require_stamp(stamp)
            result, evidence = copy.deepcopy(self.cache[key])
            self.store.emit('destination_identity.cached', {'point_id':point_id, **result, 'evidence':evidence})
        else:
            async with self.coordinator.lock:
                self.require_stamp(stamp)
                observed = await adapter.inspect('observe', {
                    'camera':point['camera_frame_id'], 'crop_xyxy':crop}, self.context.action_seq)
                self.require_stamp(stamp)
                self.context.ingest(observed)
            self.require_stamp(stamp)
            scene = observed
            sv, views = scene.state.get('views') or [], observed.state.get('views') or []
            full = next((i for i,v in enumerate(sv) if v.get('camera_frame_id') == 'agentview'), None)
            zoom = next((i for i,v in enumerate(views) if v.get('camera_frame_id') == 'zoom_preview'
                and v.get('source_camera') == point['camera_frame_id']
                and v.get('original_crop_xyxy') == crop
                and v.get('source_packet_id') == point['source_packet_id']), None)
            if (observed.state.get('error') or scene.state.get('error') or full is None or zoom is None
                    or full >= len(scene.images) or zoom >= len(observed.images)
                    or sv[full].get('source_packet_id') != point['source_packet_id']
                    or sv[full].get('is_current_observation') is not True):
                raise GateError('destination_judge_requires_exact_current_scene_and_crop')
            marked = (observed.state.get('feedback') or {}).get('point') or {}
            if any(marked.get(k) != point.get(k) for k in ('point_id','source_packet_id','camera_frame_id','pixel_xy','xyz_m')):
                raise GateError('destination_judge_requires_selected_anchor_marker; remeasure the intended destination')
            paths = [scene.images[full], observed.images[zoom]]
            evidence = {'observation_id':observed.id, 'scene_observation_id':scene.id,
                'source_packet_id':point['source_packet_id'], 'camera_frame_id':point['camera_frame_id'],
                'candidate_id':candidate_id, 'crop_xyxy':crop, 'pixel_xy':list(point['pixel_xy']),
                'xyz_m':list(point['xyz_m']), 'placement_anchor':bool(point.get('placement_anchor')),
                'action_seq':self.context.action_seq, 'control_version':self.context.control_version,
                'task_start_image_sha256':reference['image_sha256'],
                'current_image_sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]}
            query = {'mission':self.context.milestone.mission, 'control_version':self.context.control_version,
                     'evidence':evidence, 'views':[sv[full], views[zoom]],
                     'judgment_scope':'semantic_destination_at_marked_anchor_not_physical_success'}
            self.store.emit('destination_identity.started', {'point_id':point_id,'evidence':evidence})
            result = await self.model.decide(DESTINATION, query, paths)
            self.require_stamp(stamp, observed.id)
            if adapter.points.get(point_id) != point:
                raise GateError('destination_judge_point_changed')
            if (not isinstance(result,dict) or set(result) != {'verdict','selected_appearance','reason'}
                    or result['verdict'] not in {'match','mismatch','uncertain'}
                    or any(not isinstance(result[k],str) or not 1 <= len(result[k]) <= 1600
                           for k in ('selected_appearance','reason'))):
                raise ValueError('destination judge requires a bounded semantic verdict, appearance and reason')
            if result['verdict'] != 'uncertain':
                self.cache[key] = (copy.deepcopy(result),copy.deepcopy(evidence))
                if len(self.cache) > 64:
                    self.cache.pop(next(iter(self.cache)))
            self.store.emit('destination_identity.checked', {'point_id':point_id, **result, 'evidence':evidence})
        self.require_stamp(stamp)
        self.context.local_state['destination_check'] = {
            'point_id':point_id, **result, 'evidence':evidence,
            'scope':{'milestone_id':self.context.milestone.id, 'revision':self.context.milestone.revision,
                     'generation':self.context.milestone.generation, 'action_seq':self.context.action_seq,
                     'control_version':self.context.control_version,
                     'visual_scope':self.context.latest.state.get('candidate_scope')},
            'provenance':'independent_model_destination_judgment_not_world_truth',
            'readiness':('Semantic veto cleared only; physical and contract gates still apply.'
                         if result['verdict']=='match' else 'Destination not established. Re-ground the requested support/container in current images; a renamed point at the same place is not a correction.')}
        if result['verdict'] != 'match':
            self.context.local_state['readonly_streak'] += 1
            self.context.remember('destination_identity.rejected', {'point_id':point_id, **result,
                'next_step':'Identify the requested destination in the current scene and measure it. Do not repeat the rejected placement anchor or release the held object onto an unrelated surface.'})
            return False
        return True
