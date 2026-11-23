"""Slow finishes after Fast moves: historical strategy survives, authority does not."""
import asyncio
import copy
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Advice, StrategicAdvice, Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator
from test_round20_regressions import MemoryStore, FakeAdapter, QueuedModel


def harness(enabled=True):
    store = MemoryStore()
    context = Context(store)
    adapter = FakeAdapter()
    coordinator = Coordinator(adapter, context, store)
    coordinator.delegate(Milestone('m', 1, 'put the carton in the basket', 'put the carton in the basket'))
    context.ingest(Observation('o', 1, 1, time.monotonic(), 0, 0, {}))
    context.defer_routine_advice = True
    slow = SlowAgent(None, context, coordinator, store, milestone_advice_delivery=enabled)
    slow.milestones = SimpleNamespace(stamp=lambda: ('m', context.action_seq), pipelined=True,
                                      request_replan=lambda *a, **k: False)
    return store, context, adapter, coordinator, slow


def message(context, **changes):
    now = time.monotonic()
    basis = dict(observation_id='o', visual_observation_id='o', observed_at=now, published_at=now,
                 action_seq=context.action_seq, action_id=None, action_phase='idle',
                 control_version=context.control_version, ticket=1, mission=context.milestone.mission,
                 goal=context.milestone.goal, stage_index=context.milestone.stage_index)
    a = StrategicAdvice('m', 1, 1, 'advice', 'Recheck whether the carton has detached before claiming success.',
                        review_basis=basis)
    return replace(a, **changes)


@pytest.mark.parametrize('enabled', [False, True])
def test_slow_response_after_fast_moves_is_delivered_only_as_historical_strategy(enabled):
    async def run():
        store, c, _, coord, slow = harness(enabled)
        entered, release = asyncio.Event(), asyncio.Event()
        class Model:
            async def decide(self, *args):
                entered.set()
                await release.wait()
                return {'kind':'advice', 'action_sensitive':False,
                        'text':'The earlier release retained contact. Recheck current detachment evidence.'}
        slow.model = Model()
        slow.request_review('release uncertain')
        before = c.control_version
        task = asyncio.create_task(slow.review())
        await entered.wait()
        c.action_seq = 7  # A strategy is not discarded merely after four more actions.
        c.ingest(Observation('new', 2, 1, time.monotonic(), 1, 7, {}))
        release.set()
        await task
        rows = c.projection('fast')['advice']
        assert bool(rows) is enabled
        assert c.control_version == before and not coord.paused
        if enabled:
            assert rows[0]['scope'] == 'milestone_strategy'
            assert rows[0]['age_actions'] == 7
            assert rows[0]['review_basis']['observation_id'] == 'o'
            assert rows[0]['authority'] == 'hypothesis_requiring_current_recheck'
            assert 'CURRENT' in rows[0]['instruction']
            assert slow.delivered_ticket == slow.request_ticket == 1
            assert any(e['kind']=='slow.review_finished' and e['payload']['outcome']=='strategy_advice' for e in store.events)
        else:
            assert slow.delivered_ticket == 0
    asyncio.run(run())


def test_old_holding_claim_is_never_projected_as_live_fact_or_authority():
    _, c, _, coord, _ = harness()
    a = message(c, text='The bowl was held in the review. Continue placing the held bowl.')
    c.action_seq += 1
    assert coord.apply_advice(a)
    row = c.projection('fast')['advice'][0]
    assert row['status'] == 'historical_review_not_current_state'
    assert 'not a current pose, held-object or completion fact' in row['instruction']
    assert not c.target_plan and not c.stage_evidence
    assert c.local_state['last_action'] is None


def test_delayed_strategy_survives_full_stale_operational_queue():
    store, c, _, coord, _ = harness()
    for i in range(4):
        assert coord.apply_advice(Advice('m',1,1,'advice',f'old {i}',time.monotonic()+180,
                                        expected_action_seq=0))
    a = message(c)
    c.action_seq = 1
    assert coord.apply_advice(a)
    assert list(c.strategic_advice) == [a.id]
    assert c.projection('fast')['advice'][0]['age_actions'] == 1
    assert sum(e['kind']=='advice.strategy_delivered' for e in store.events) == 1


def test_full_live_operational_queue_does_not_evict_historical_message():
    store, c, _, coord, _ = harness()
    for i in range(4):
        assert coord.apply_advice(Advice('m',1,1,'advice',f'live {i}',time.monotonic()+180,
                                        expected_action_seq=0))
    a = message(c)
    assert coord.apply_advice(a)
    assert len(c.advice) == 4 and list(c.strategic_advice) == [a.id]
    assert len(c.projection('fast')['advice']) == 5


@pytest.mark.parametrize('kind', ['pause', 'replan', 'delegate'])
def test_strategy_cannot_carry_control_authority(kind):
    _, c, _, coord, _ = harness()
    before = c.control_version
    assert not coord.apply_advice(message(c, kind=kind))
    assert not coord.paused and c.control_version == before and not c.advice


@pytest.mark.parametrize('field,value', [('expected_action_seq',0),('expected_action_phase','running'),
    ('expected_action_id','action'),('expected_visual_scope','oldframe')])
def test_strategy_cannot_relabel_a_current_action_or_pixel_binding(field, value):
    _, c, _, coord, _ = harness()
    assert not coord.apply_advice(message(c, **{field:value}))


@pytest.mark.parametrize('text', ['Use p33o-c2.', 'Use point-9.', 'At pixels 30, 40.',
    'Move x=0.12.', 'Use [0.1, 0.2, 0.3].', 'Move up 0.05 m.', 'move_to(xyz_m=[0,0,1])'])
def test_frame_specific_or_metric_text_keeps_original_action_boundary(text):
    _, c, _, coord, slow = harness()
    assert slow.strategy_message(text, c.milestone, message(c).review_basis) is None
    assert not coord.apply_advice(message(c, text=text))


@pytest.mark.parametrize('change', ['generation', 'mission', 'reverse_action'])
def test_task_boundary_or_invalid_provenance_removes_history(change):
    store, c, _, coord, _ = harness()
    a = message(c)
    assert coord.apply_advice(a)
    if change == 'generation': c.latest = replace(c.latest, generation=2)
    elif change == 'mission': c.milestone = replace(c.milestone, mission='another user task')
    else: c.strategic_advice[a.id]['review_basis']['action_seq'] = 2
    assert not c.projection('fast')['advice']
    assert any(e['kind']=='advice.strategy_invalid' for e in store.events)


def test_old_review_has_no_expiry_and_does_not_rejuvenate_evidence():
    _, c, _, coord, slow = harness()
    basis = message(c).review_basis
    basis['observed_at'] -= 3600
    basis['published_at'] = time.monotonic()+500
    a = slow.strategy_message('Recheck detachment against current contact.',c.milestone,basis)
    assert a.review_basis['observed_at'] == basis['observed_at']
    assert a.review_basis['published_at'] < basis['published_at']
    assert a.expires_at is None
    c.action_seq += 8
    assert coord.apply_advice(a)
    row = c.projection('fast')['advice'][0]
    assert row['age_s'] >= 3600 and row['age_actions'] == 8
    assert row['status'] == 'historical_review_not_current_state'


@pytest.mark.parametrize('change', ['missing_publication','future_publication','future_observation','finite_expiry'])
def test_message_provenance_and_no_expiry_envelope_are_validated(change):
    _, c, _, coord, _ = harness()
    a = message(c)
    if change == 'missing_publication': del a.review_basis['published_at']
    elif change == 'future_publication': a.review_basis['published_at'] += 10
    elif change == 'future_observation': a.review_basis['observed_at'] += 10
    else: a = replace(a,expires_at=time.monotonic()+120)
    assert not coord.apply_advice(a)


def test_normal_action_advice_and_pause_remain_stale_after_action_progress():
    store, c, _, coord, _ = harness()
    for kind in ['advice','pause']:
        a = Advice('m',1,1,kind,'old current-action statement',time.monotonic()+120,expected_action_seq=0)
        c.action_seq=1
        assert not coord.apply_advice(a)
    assert not coord.paused and not c.advice


def test_same_action_finishing_does_not_discard_strategy_but_discards_operational_advice():
    _, c, _, coord, _ = harness()
    historical = message(c)
    historical.review_basis.update(action_phase='running', action_id='done')
    operational = Advice('m',1,1,'advice','current movement',time.monotonic()+120,
                         expected_action_seq=0,expected_action_id='done',expected_action_phase='running')
    assert coord.running is None
    assert coord.apply_advice(historical)
    assert not coord.apply_advice(operational)


def test_strategy_is_seen_only_on_next_fast_decision_without_extra_call_or_preemption():
    async def run():
        store, c, _, coord, _ = harness()
        entered, release = asyncio.Event(), asyncio.Event()
        class Model:
            def __init__(self): self.calls=[]
            async def decide(self, prompt, projection, images):
                self.calls.append(copy.deepcopy(projection))
                if len(self.calls)==1:
                    entered.set()
                    await release.wait()
                return {'kind':'tool','tool':'move_to','args':{},'advice_response':'Rechecked against current contact.'}
        model=Model()
        fast=FastAgent(model,c,coord,store)
        before=c.control_version
        task=asyncio.create_task(fast.run(2))
        await entered.wait()
        assert coord.apply_advice(message(c))
        assert c.control_version==before and not task.done()
        release.set()
        assert await task=='decision_budget'
        assert len(model.calls)==2
        assert model.calls[0]['advice']==[]
        assert len(model.calls[1]['advice'])==1
        assert model.calls[1]['advice'][0]['age_actions']==1
        assert not any(e['kind'] in {'fast.decision_stale','fast.decision_preempted'} for e in store.events)
        assert any(e['kind']=='advice.considered' for e in store.events)
    asyncio.run(run())


def test_ordinary_advice_serialization_is_unchanged_when_feature_is_off():
    assert set(asdict(Advice('m',1,1,'advice','text',1))) == {
        'milestone_id','revision','generation','kind','text','expires_at',
        'expected_action_seq','expected_action_id','expected_action_phase','id','expected_visual_scope','priority'}
