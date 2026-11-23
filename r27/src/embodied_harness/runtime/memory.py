"""Explicitly enabled, verified prior-episode appearance memory, never current poses."""
import hashlib
import json
import re
from pathlib import Path


def _validate_acquisition_reference(entry, image):
    """Check frozen experience evidence without consulting a live evaluator."""
    p = entry.get('provenance', {})
    if (p.get('official_success') is not True or type(p.get('accepted_grasp_count')) is not int
            or p['accepted_grasp_count'] != 1 or type(p.get('pregrasp_action_seq')) is not int
            or p['pregrasp_action_seq'] != 0 or p.get('receipt_status') != 'completed'
            or p.get('receipt_macro_status') != 'lift_completed_verify_retention_visually'):
        raise ValueError('Appearance experience requires one completed acquisition before official success')
    seqs = [p.get(k) for k in ('accepted_grasp_seq', 'receipt_seq', 'official_verification_seq')]
    if any(type(x) is not int for x in seqs) or not 0 < seqs[0] < seqs[1] < seqs[2]:
        raise ValueError('Appearance experience evidence order is invalid')
    verified = p.get('official_verification', {})
    if verified.get('status') != 'verified' or verified.get('source') != 'official_libero_check_success':
        raise ValueError('Appearance experience requires an official verification record')
    for key in ('source_result_sha256', 'source_events_sha256', 'source_fast_trace_sha256',
                'source_image_sha256'):
        if not isinstance(p.get(key), str) or re.fullmatch(r'[0-9a-f]{64}', p[key]) is None:
            raise ValueError('Appearance experience requires immutable source hashes')
    expected = entry.get('image_sha256')
    if (not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None
            or hashlib.sha256(image.read_bytes()).hexdigest() != expected
            or p.get('reference_image_sha256') != expected):
        raise ValueError('Appearance reference image differs from its recorded source')
    if p.get('reference_transform') != {'method':'strip_debug_margins_v1',
                                        'source_size':[512,512], 'crop_xyxy':[48,16,512,474]}:
        raise ValueError('Appearance reference must remove historical candidate IDs and coordinate overlays')


def retrieve(path,mission, *, appearance_reference_memory=False):
    if type(appearance_reference_memory) is not bool:
        raise ValueError('appearance_reference_memory must be boolean')
    if not path:return []
    manifest=Path(path).resolve();entries=json.loads(manifest.read_text())['entries'];matches=[]
    for entry in entries:
        memory_class = entry.get('memory_class')
        if memory_class is not None:
            if not appearance_reference_memory: continue
            if memory_class != 'verified_single_acquisition_v1':
                raise ValueError('Unknown appearance experience class')
            phrase = entry.get('match_text')
            if not isinstance(phrase, str) or not phrase.strip():
                raise ValueError('Appearance experience requires a nonempty target phrase')
            if re.search(r'(?<!\w)' + re.escape(phrase.casefold()) + r'(?!\w)', mission.casefold()) is None:
                continue
        elif entry['match_text'].casefold() not in mission.casefold():continue
        if entry['provenance'].get('official_success') is not True:
            raise ValueError('Appearance memory needs a verified source episode')
        image=(manifest.parent/entry['image']).resolve()
        if not image.is_relative_to(manifest.parent):raise ValueError('Memory image must be within manifest directory')
        if memory_class is not None:
            _validate_acquisition_reference(entry, image)
        matches.append({**entry,'image':str(image),'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
    return matches[:2]
