"""Frozen evaluation-only English ambiguity cases.

Gold/review notes must NEVER enter prompts or model requests. User-submitted
examples are unchanged, including spelling and spaces. Seven user examples
and thirty newly composed controls; none of the previous ten benchmark inputs
are used to author thresholds.

An out-of-catalog website context is a TEST FIXTURE, not a database row.
The Thinker sees only the original phrase and the same 20-category dictionary;
only the Selector may see the actual website domain.
"""

from dataclasses import dataclass

from scripts.inspect_english_senses import CASES, EXTENDED
from scripts.inspect_context_selector import REVIEW_TARGETS

# Unpublished test-site domains: static evaluation metadata, NEVER appended to
# the Thinker's 20-context dictionary or written into PostgreSQL.
EXTERNAL_SITE_CONTEXTS = {
    'maritime': {
        'title': 'Maritime operations',
        'description': 'Seafaring, ships, hull maintenance and vessel motion.',
    },
    'wildlife': {
        'title': 'Wildlife and zoology',
        'description': 'Animals, species identification and natural history.',
    },
    'astronomy': {
        'title': 'Astronomy and night sky',
        'description': 'The Moon, planetary science and observations of the sky.',
    },
}


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    site_context_id: str
    review_target: str
    group: str
    review_status: str = 'gold'  # gold or qualitative (no forced binary grade)


# Keep the originally supplied spelling/casing intact.
USER_CASES = (
    Case('u_drop_base', 'Drop the base', 'music',
         "Do NOT silently change base to bass; music context alone is insufficient "
         "to assume an EDM 'drop the bass' command.", 'user', 'qualitative'),
    Case('u_blockbuster', 'Blockbuster', 'military',
         'Large high-explosive aerial bomb (military-era sense), not film hit.', 'user'),
    Case('u_moon_shine', 'Moon shine', 'astronomy',
         'Preserve the two-word spelling; literal shining moon/moonlight is plausible; '
         'do not silently collapse to moonshine liquor.', 'user', 'qualitative'),
    Case('u_honey_badger', 'Honey badger', 'wildlife',
         'Animal: honey badger/ratel; not honey or a badger separately.', 'user'),
    Case('u_heaving_down', 'Heaving down', 'maritime',
         'Ship careening: deliberately heeling a vessel to expose underwater hull.', 'user'),
    Case('u_pitching', 'pitching', 'maritime',
         'Ship pitching: oscillatory bow-up and bow-down rotation about transverse axis.', 'user'),
    Case('u_go_bear', 'go to a bear', 'finance',
         'Unverified as a fixed trading idiom: do not confidently invent '
         'a short-selling meaning from the noun bear alone.', 'user', 'qualitative'),
)

# Fresh controls, phrased independently of the known ten regression cases.
AUTHOR_CASES = (
    Case('a_charge_dismissed', 'The charge was dismissed.', 'legal',
         'Legal accusation or criminal charge was dismissed.', 'author'),
    Case('a_charge_measured', 'The charge was measured.', 'electronics',
         'Electrical charge quantity was measured.', 'author'),
    Case('a_patient_discharge', 'The patient was discharged.', 'healthcare',
         'Patient was released from medical care.', 'author'),
    Case('a_capacitor_discharge', 'The capacitor discharged.', 'electronics',
         'Stored electric charge was released.', 'author'),
    Case('a_pitch_rose', 'The pitch rose sharply.', 'music',
         'Musical pitch/frequency rose.', 'author'),
    Case('a_pitch_inspected', 'The pitch was inspected before kickoff.', 'sports',
         'Playing field was inspected before a sports match.', 'author'),
    Case('a_current_increased', 'The current increased.', 'electronics',
         'Electrical current rose.', 'author'),
    Case('a_current_canoe', 'The current carried the canoe downstream.', 'electronics',
         'Flowing water in a river, despite electronics context.', 'author'),
    Case('a_bond_matured', 'The bond matured.', 'finance',
         'Debt security reached maturity.', 'author'),
    Case('a_bond_between', 'The bond between them grew stronger.', 'finance',
         'Interpersonal connection became stronger despite finance context.', 'author'),
    Case('a_bank_collapsed', 'The bank collapsed.', 'finance',
         'Financial institution failed; absent disambiguating source cues.', 'author'),
    Case('a_river_bank', 'The river bank collapsed.', 'finance',
         'Riverbank gave way, not financial institution.', 'author'),
    Case('a_driver_crashed', 'The driver crashed.', 'software',
         'Software/device driver stopped functioning, context resolves ambiguity.', 'author'),
    Case('a_driver_truck', 'The driver parked the truck.', 'software',
         'Human vehicle driver, despite software context.', 'author'),
    Case('a_cell_divided', 'The cell divided.', 'healthcare',
         'Biological cell divided.', 'author'),
    Case('a_cell_locked', 'The cell was locked.', 'technology',
         'Likely a locked room or confinement cell; no automatic biological/electrical sense.', 'author'),
    Case('a_crane_lifted', 'The crane lifted the beam.', 'construction',
         'Mechanical construction crane hoisted a beam.', 'author'),
    Case('a_crane_wings', 'The crane spread its wings.', 'construction',
         'Bird crane spread wings, despite construction context.', 'author'),
    Case('a_note_four_beats', 'The note was held for four beats.', 'music',
         'A musical note sustained for four beats.', 'author'),
    Case('a_note_invoice', 'The note was added to the invoice.', 'music',
         'Written annotation added to an invoice, despite music context.', 'author'),
    Case('a_port_blocked', 'The port was blocked.', 'computing_networks',
         'Network/communication port blocked, disambiguated by domain.', 'author'),
    Case('a_harbor_port', 'The harbor port was closed.', 'computing_networks',
         'Seaport or harbor terminal closed; not a network port.', 'author'),
    Case('a_needle_vein', 'The needle entered the vein.', 'healthcare',
         'Medical needle inserted into vein.', 'author'),
    Case('a_needle_record', 'The needle skipped across the record.', 'healthcare',
         'Record-player stylus moved across a vinyl record, not a medical needle.', 'author'),
    Case('a_battery_died', 'The battery died.', 'electronics',
         'Battery lost charge or ability to supply power.', 'author'),
    Case('a_battery_fire', 'The battery opened fire.', 'electronics',
         'Artillery battery began firing weapons; not electrochemical battery.', 'author'),
    Case('a_ticket_court', 'The ticket was dismissed by the court.', 'events_ticketing',
         'Traffic/legal citation dismissed, not event admission ticket.', 'author'),
    Case('a_shot_goalkeeper', 'The shot was blocked by the goalkeeper.', 'healthcare',
         'Sport shot toward goal blocked by goalkeeper, not an injection.', 'author'),
    Case('a_shot_clinic', 'The shot was given in the clinic.', 'sports',
         'Injection or vaccination administered at clinic, not a sports attempt.', 'author'),
    Case('a_contract_executed', 'The contract was executed.', 'legal',
         'A contract was signed or legally completed; not computer code execution.', 'author'),
)

FRESH_CASES = USER_CASES + AUTHOR_CASES

REGRESSION_CASES = tuple(
    Case(case_id, source, context, REVIEW_TARGETS[case_id], 'regression')
    for case_id, source, context in CASES + EXTENDED
)


def validate_cases() -> None:
    assert len(USER_CASES) == 7, 'Seven user-contributed cases expected'
    assert len(AUTHOR_CASES) == 30, 'Thirty independent controls expected'
    assert len(FRESH_CASES) == 37, 'Expected 37 unseen test cases'
    assert len(REGRESSION_CASES) == 10
    ids = [case.case_id for case in FRESH_CASES + REGRESSION_CASES]
    assert len(ids) == len(set(ids))
    assert all(case.source.strip() and case.review_target for case in FRESH_CASES)
    assert all(case.review_status in ('gold', 'qualitative') for case in FRESH_CASES)
    assert {'maritime', 'wildlife', 'astronomy'} <= set(EXTERNAL_SITE_CONTEXTS)


validate_cases()
