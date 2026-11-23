"""Persistent messages are information; current control remains authoritative."""
import asyncio
import json
import time
from dataclasses import replace

from embodied_harness.runtime.milestones import MilestoneManager
from embodied_harness.domain.models import Advice, Observation
from embodied_harness.runtime.coordinator import GateError
from test_round25_advice_delivery import harness, message


def test_all_40_messages_survive_time_actions_pause_and_contract_handoff(monkeypatch):
    store, c, _, coord, slow = harness()
    messages = [message(c,text=f'Historical diagnostic number {i}') for i in range(40)]
    for a in messages: assert coord.apply_advice(a)
    now = time.monotonic()
    monkeypatch.setattr('time.monotonic',lambda:now+3600)
    c.action_seq = 100
    c.ingest(Observation('later',2,1,now+3600,10,100,{'contact_state':{'measurement_generation':50}}))
    coord.pause('new external stop')
    version=c.control_version
    coord.mission_authority={'mission':c.milestone.mission,'capabilities':frozenset(c.milestone.capabilities),
                             'resources':frozenset(c.milestone.resources),'constraints':frozenset()}
    manager=MilestoneManager(slow,c,coord,store,24,now+7200)
    manager.stages=[{'goal':'now complete the placement','success_conditions':[]}]
    manager.plan_id='next-plan'
    assert manager.activate(0)
    for role in ['fast','slow']:
        rows=c.projection(role)['advice']
        assert len(rows)==40
        assert [r['id'] for r in rows]==[a.id for a in messages]
        assert all(r['age_s']>=3600 and r['since_publication_s']>=3600 and r['age_actions']==100 for r in rows)
        assert all(r['expires_at'] is None for r in rows)
        assert all(r['relation_to_current']=={'same_milestone':False,'same_control_version':False} for r in rows)
        assert all(r['review_basis']['goal']=='put the carton in the basket' for r in rows)
        json.dumps(rows,allow_nan=False)
    assert coord.paused and c.control_version==version+1  # Only contract delegation, no message signal.
    assert not any(e['kind']=='advice.strategy_expired' for e in store.events)


def test_late_previous_stage_review_is_kept_as_old_stage_information():
    _, c, _, coord, slow = harness()
    old=c.milestone
    basis=message(c).review_basis
    coord.delegate(replace(old,id='new-stage',goal='later task stage'),expected_revision=old.revision)
    a=slow.strategy_message('Reconsider the earlier diagnostic against current images.',old,basis)
    assert coord.apply_advice(a)
    row=c.projection('fast')['advice'][0]
    assert row['relation_to_current']=={'same_milestone':False,'same_control_version':False}
    assert row['milestone_id']==old.id and row['review_basis']['goal']==old.goal
    assert c.milestone.id=='new-stage'


def test_old_episode_review_cannot_leak_into_new_generation():
    _, c, adapter, coord, slow = harness()
    old=c.milestone
    a=message(c)
    assert coord.apply_advice(a)
    adapter.generation=2
    c.milestone=replace(old,generation=2)
    c.ingest(Observation('reset',1,2,time.monotonic(),0,0,{}))
    assert not c.projection('fast')['advice']
    assert not c.strategic_advice
    late=slow.strategy_message('Earlier task evidence is historical.',old,a.review_basis)
    assert not coord.apply_advice(late)


def test_strategy_id_is_idempotent_and_collision_does_not_overwrite():
    store, c, _, coord, _ = harness()
    a=message(c)
    assert coord.apply_advice(a) and coord.apply_advice(a)
    assert not coord.apply_advice(replace(a,text='Conflicting replacement'))
    assert len(c.strategic_advice)==1 and c.strategic_advice[a.id]['text']==a.text
    assert sum(e['kind']=='advice.strategy_delivered' for e in store.events)==1


def test_equal_publication_times_preserve_arrival_order(monkeypatch):
    _, c, _, coord, _ = harness()
    now=time.monotonic()
    monkeypatch.setattr('time.monotonic',lambda:now)
    for ident in ['z-first','a-second']:
        assert coord.apply_advice(message(c,id=ident))
    assert [r['id'] for r in c.projection('fast')['advice']]==['z-first','a-second']


def test_expired_operational_advice_still_disappears_next_to_history(monkeypatch):
    _, c, _, coord, _ = harness()
    now=time.monotonic()
    assert coord.apply_advice(Advice('m',1,1,'advice','current operation',now+10,expected_action_seq=0))
    a=message(c)
    assert coord.apply_advice(a)
    monkeypatch.setattr('time.monotonic',lambda:now+7200)
    rows=c.projection('fast')['advice']
    assert [r['id'] for r in rows]==[a.id]
    assert not c.advice


def test_ordinary_queue_can_receive_after_history_without_null_expiry_error():
    _, c, _, coord, _ = harness()
    a=message(c)
    assert coord.apply_advice(a)
    assert coord.apply_advice(Advice('m',1,1,'advice','new operational note',time.monotonic()+10))
    assert len(c.projection('fast')['advice'])==2
