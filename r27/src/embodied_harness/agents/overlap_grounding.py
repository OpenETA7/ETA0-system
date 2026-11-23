"""Speculate only a semantic veto; no result is authority to execute a motion."""
import asyncio
import copy
import hashlib
import time
from pathlib import Path

from .grounding import MissionTargetJudge, TARGET_IDENTITY
from embodied_harness.runtime.coordinator import GateError


class OverlappedTargetJudge(MissionTargetJudge):
    """One bounded inference concurrent with the actor's next decision.

    Capture is read-only and does not replace the actor's observation. A result
    is published only when the actor actually asks to grasp the same point under
    the same observation, physical revision, contract and control state. Other
    choices cancel/discard the work, including already returned results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = None
        self.serial = 0

    def key_for(self, point, stamp, reference):
        visual = point.get('visual_candidate') or {}
        x, y = point['pixel_xy']
        bbox = visual.get('bbox_xyxy')
        crop = ([max(0,bbox[0]-8),max(0,bbox[1]-8),min(512,bbox[2]+8),min(512,bbox[3]+8)]
                if bbox else [max(0,x-35),max(0,y-35),min(512,x+36),min(512,y+36)])
        return ((stamp, reference['image_sha256'], point['source_packet_id'],
                 point['camera_frame_id'], visual.get('candidate_id'), tuple(crop)), crop)

    async def discard(self, reason='unused_actor_choice'):
        pending, self.pending = self.pending, None
        if not pending:
            return
        task = pending['task']
        was_pending = not task.done()
        if was_pending:
            task.cancel()
        try:
            await task
            outcome = 'returned_but_unused'
        except asyncio.CancelledError:
            outcome = 'cancelled'
        except Exception as exc:
            outcome = 'failed_' + type(exc).__name__
        self.store.emit('target_identity.prefetch_discarded', {
            'prefetch_id':pending['id'], 'point_id':pending['point_id'], 'reason':reason,
            'outcome':outcome, 'elapsed_s':time.monotonic()-pending['started'],
            'provider_cost':'reported usage only; cancellation does not prove unbilled'})
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError  # Never turn caller shutdown into another actor decision.

    async def prefetch_current(self):
        await self.discard('next_decision')
        local, obs = self.context.local_state, self.context.latest
        point_id = ((obs.state.get('feedback') or {}).get('point') or {}).get('point_id') if obs else None
        if (not point_id or getattr(self.coordinator.adapter, 'possible_hold', None) is not False
                or local.get('possible_hold') is True
                or local.get('preparation_invalidated')):
            return
        try:
            stamp = self.stamp()
            reference = getattr(self.model, 'mission_reference_metadata', None)
            if not reference or not reference.get('image_sha256'):
                return
            point = copy.deepcopy(self.coordinator.adapter.points[point_id])
            if point.get('camera_frame_id') != 'agentview' or not point.get('visual_candidate'):
                return  # Noncanonical/manual points retain the ordinary mandatory check.
            key, crop = self.key_for(point, stamp, reference)
            if key in self.cache:
                return
            # This validation is read-only; the ordinary execution admission still
            # runs after the actual decision. It prevents speculative stale points.
            self.coordinator.adapter.validate_args('grasp_point', {'point_id':point_id})
            async with self.coordinator.lock:
                self.require_stamp(stamp, obs.id)
                captured = await self.coordinator.adapter.inspect('observe', {
                    'camera':'agentview', 'crop_xyxy':crop}, self.context.action_seq)
                self.require_stamp(stamp, obs.id)
            before_sensor, after_sensor = obs.state.get('sensor') or {}, captured.state.get('sensor') or {}
            physical = lambda sensor: {k:v for k,v in sensor.items() if k not in {'captured_at','visual_sample'}}
            if (not all(k in before_sensor and k in after_sensor for k in ('sim_time','control_steps'))
                    or physical(before_sensor) != physical(after_sensor)
                    or obs.state.get('robot') != captured.state.get('robot')
                    or obs.state.get('tactile') != captured.state.get('tactile')):
                raise GateError('prefetch_capture_changed_physical_evidence')
            views = captured.state.get('views') or []
            full = next((i for i,v in enumerate(views) if v.get('camera_frame_id') == 'agentview'), None)
            zoom = next((i for i,v in enumerate(views) if v.get('camera_frame_id') == 'zoom_preview'
                and v.get('source_camera') == 'agentview' and v.get('original_crop_xyxy') == crop
                and v.get('source_packet_id') == point['source_packet_id']), None)
            if (captured.state.get('error') or full is None or zoom is None
                    or full >= len(captured.images) or zoom >= len(captured.images)
                    or captured.action_seq != self.context.action_seq
                    or captured.generation != self.coordinator.adapter.generation
                    or captured.state.get('candidate_scope') != obs.state.get('candidate_scope')
                    or views[full].get('source_packet_id') != point['source_packet_id']
                    or views[full].get('is_current_observation') is not True):
                raise GateError('prefetch_requires_exact_current_scene_and_crop')
            paths = [captured.images[full], captured.images[zoom]]
            evidence = {'observation_id':captured.id, 'scene_observation_id':captured.id,
                'source_packet_id':point['source_packet_id'], 'camera_frame_id':'agentview',
                'candidate_id':point['visual_candidate'].get('candidate_id'), 'crop_xyxy':crop,
                'action_seq':self.context.action_seq, 'control_version':self.context.control_version,
                'task_start_image_sha256':reference['image_sha256'],
                'current_image_sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]}
            query = {'mission':self.context.milestone.mission, 'control_version':self.context.control_version,
                     'evidence':evidence, 'views':[views[full],views[zoom]]}
            self.serial += 1
            pending = {'id':self.serial, 'point_id':point_id, 'point':point, 'stamp':stamp,
                       'anchor_id':obs.id, 'key':key, 'evidence':evidence, 'started':time.monotonic()}
            self.store.emit('target_identity.prefetch_started', {
                'prefetch_id':self.serial, 'point_id':point_id, 'anchor_observation_id':obs.id,
                'evidence':evidence, 'scope':list(stamp), 'actor_observation_unchanged':True})
            self.store.emit('target_identity.started', {'point_id':point_id,'evidence':evidence,
                                                       'prefetch_id':self.serial})
            pending['task'] = asyncio.create_task(self.infer(pending, query, paths))
            self.pending = pending
        except Exception as exc:
            # Missing speculative evidence never waives the mandatory pre-grasp
            # check. The actual request follows the ordinary path or is rejected.
            self.store.emit('target_identity.prefetch_skipped', {'point_id':point_id,
                'reason':type(exc).__name__, 'mandatory_check_unchanged':True})

    async def infer(self, pending, query, paths):
        result = await self.model.decide(TARGET_IDENTITY, query, paths)
        self.require_stamp(pending['stamp'], pending['anchor_id'])
        if (not isinstance(result,dict) or set(result) != {'verdict','selected_appearance','reason'}
                or result['verdict'] not in {'match','mismatch','uncertain'}
                or any(not isinstance(result[k],str) or not 1 <= len(result[k]) <= 1600
                       for k in ('selected_appearance','reason'))):
            raise ValueError('target judge requires a bounded semantic verdict, appearance and reason')
        self.store.emit('target_identity.prefetch_returned', {'prefetch_id':pending['id'],
            'elapsed_s':time.monotonic()-pending['started'], 'verdict':result['verdict'],
            'published_to_actor':False})
        return result

    async def check(self, point_id):
        pending = self.pending
        if not pending or pending['point_id'] != point_id:
            await self.discard('different_grasp_point')
            self.store.emit('target_identity.prefetch_fallback', {'point_id':point_id})
            return await super().check(point_id)
        started = time.monotonic()
        try:
            self.require_stamp(pending['stamp'], pending['anchor_id'])
            if self.coordinator.adapter.points.get(point_id) != pending['point']:
                raise GateError('prefetched_point_changed')
            result = await pending['task']
            self.require_stamp(pending['stamp'], pending['anchor_id'])
            if self.coordinator.adapter.points.get(point_id) != pending['point']:
                raise GateError('prefetched_point_changed')
            self.pending = None
            self.cache[pending['key']] = (copy.deepcopy(result), copy.deepcopy(pending['evidence']))
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
            self.store.emit('target_identity.checked', {'point_id':point_id, **result,
                'evidence':pending['evidence'], 'prefetch_id':pending['id']})
            self.store.emit('target_identity.prefetch_consumed', {'prefetch_id':pending['id'],
                'point_id':point_id, 'blocking_wait_s':time.monotonic()-started,
                'elapsed_s':time.monotonic()-pending['started']})
            try:
                return await super().check(point_id)
            finally:
                if result['verdict'] == 'uncertain':
                    self.cache.pop(pending['key'], None)  # Same baseline uncertain-cache policy.
        except BaseException:
            await self.discard('consumption_failed_or_stale')
            raise
