"""Read-only, experiment-only overlay for the 20-context Thinker trial.

The six previously published contexts are read from PostgreSQL as usual. The
fourteen definitions below exist ONLY in memory while the benchmark runs.
This module does not create tables, publish profiles, or write database rows.
"""

from copy import deepcopy


# Stable IDs for reporting. Definitions describe domains broadly rather than
# naming any of the benchmark's English test sentences or expected answers.
ADDITIONAL_CONTEXTS = {
    'computing_networks': {
        'title': 'Computing and Networks',
        'description': 'Computer architecture, digital communication and networking.',
        'specialization': 'computer systems and networking',
        'scope': (
            'Computer interconnects, data transfer, communications protocols, '
            'network equipment, infrastructure, distributed systems, and diagnostics.'
        ),
    },
    'road_transport': {
        'title': 'Road Transport',
        'description': 'Road vehicles, public mobility and road logistics.',
        'specialization': 'road passenger and freight transportation',
        'scope': (
            'Road vehicles, urban and intercity passenger services, freight '
            'operations, traffic infrastructure, fleet management and maintenance.'
        ),
    },
    'finance': {
        'title': 'Finance',
        'description': 'Banking, payments, investments and financial instruments.',
        'specialization': 'banking and financial services',
        'scope': (
            'Payments, currency, lending, accounting, financial markets, '
            'securities, instruments, and reporting.'
        ),
    },
    'legal': {
        'title': 'Law and Legal Services',
        'description': 'Legal terminology, proceedings and contracts.',
        'specialization': 'legal practice and regulation',
        'scope': (
            'Contracts, courts, legislation, legal obligations, disputes, '
            'formal notices, and regulatory compliance.'
        ),
    },
    'healthcare': {
        'title': 'Healthcare',
        'description': 'Medical services, patient care and clinical practice.',
        'specialization': 'healthcare and medicine',
        'scope': (
            'Clinical records, diagnostics, therapies, hospital operations, '
            'medical procedures, pharmaceuticals and patient communications.'
        ),
    },
    'education': {
        'title': 'Education',
        'description': 'Teaching, learning, assessment and academic programs.',
        'specialization': 'education and academic administration',
        'scope': (
            'Schools, universities, educational programs, examinations, '
            'coursework, instruction, grading and educational administration.'
        ),
    },
    'hospitality': {
        'title': 'Hospitality',
        'description': 'Accommodation, lodging and guest services.',
        'specialization': 'hospitality and lodging',
        'scope': (
            'Hotels, accommodation, guest check-in, reservations, visitor '
            'amenities, and lodging operations.'
        ),
    },
    'office_administration': {
        'title': 'Office Administration',
        'description': 'Business offices, clerical work and visitor reception.',
        'specialization': 'office and administrative services',
        'scope': (
            'Office operations, document handling, reception services, '
            'administrative communications, scheduling and clerical work.'
        ),
    },
    'events_ticketing': {
        'title': 'Events and Admissions',
        'description': 'Public events, admission and venue operations.',
        'specialization': 'events, admissions and venues',
        'scope': (
            'Public events, audience access, ticket sales and validation, '
            'venue organization, admission policies and event operations.'
        ),
    },
    'food_service': {
        'title': 'Food Service',
        'description': 'Restaurants, commercial kitchens and dining.',
        'specialization': 'restaurant and catering operations',
        'scope': (
            'Dining service, restaurants, commercial food preparation, '
            'menus, waiting staff, catering and kitchen procedures.'
        ),
    },
    'publishing': {
        'title': 'Publishing',
        'description': 'Written works, editing, editorial production and print.',
        'specialization': 'publishing and editorial work',
        'scope': (
            'Manuscripts, editorial annotation, literature, newspapers, '
            'academic publishing, typesetting, and publication workflows.'
        ),
    },
    'sports': {
        'title': 'Sports',
        'description': 'Athletics, sports competitions and training.',
        'specialization': 'sport and athletic events',
        'scope': (
            'Athletics, sporting facilities, races, competitive teams, '
            'rules, training activities, results and event organization.'
        ),
    },
    'construction': {
        'title': 'Construction',
        'description': 'Buildings, civil works and structural engineering.',
        'specialization': 'construction and civil engineering',
        'scope': (
            'Building sites, foundations, structures, construction projects, '
            'infrastructure works, engineering materials and site safety.'
        ),
    },
    'land_management': {
        'title': 'Land and Environment',
        'description': 'Land use, outdoor infrastructure and natural environments.',
        'specialization': 'land management and environmental services',
        'scope': (
            'Parks, land use, outdoor paths, habitats, environmental protection, '
            'terrain maintenance and conservation work.'
        ),
    },
}

ORIGINAL_CONTEXT_IDS = {
    'technology', 'electronics', 'software', 'railway', 'music', 'military',
}


def expand_to_20(published_catalog: dict) -> dict:
    """Create an isolated 20-entry catalog without modifying the input or DB.

    Fail closed if the published catalog has drifted; the test must not silently
    become 21+ contexts or overwrite a real published context.
    """
    existing = set(published_catalog)
    if existing != ORIGINAL_CONTEXT_IDS:
        raise ValueError(
            '20-context experiment expects six published contexts; got '
            + ', '.join(sorted(existing))
        )
    if existing & set(ADDITIONAL_CONTEXTS):
        raise ValueError('Experiment-only context collides with published ID')
    if len(ADDITIONAL_CONTEXTS) != 14:
        raise ValueError('Expected fourteen experiment-only contexts')

    result = deepcopy(published_catalog)
    for context_id, item in ADDITIONAL_CONTEXTS.items():
        result[context_id] = {
            'title': item['title'],
            'description': item['description'],
            'version': 'experiment-only',
            'experiment_only': True,
            'profiles_general_to_specific': [{
                'specialization': item['specialization'],
                'scope': item['scope'],
            }],
        }
    if len(result) != 20:
        raise ValueError('Expanded context catalog must contain exactly 20 IDs')
    return dict(sorted(result.items()))
