"""Default profile presets for csar2App.

Profiles are intentionally conservative and modular so new satellites can be
added by extending this dictionary without touching the processing app itself.
"""

CSAR2_PROFILES = {
    'LT1': {
        'sensor': 'LUTAN1',
        'description': 'Lutan-1 stripmap high-resolution profile',
    },
    'GF3': {
        'sensor': 'GF3',
        'description': 'Gaofen-3 stripmap profile (GF3 adapter)',
    },
    'DJ1': {
        'sensor': 'DJ1',
        'description': 'DJ-1 stripmap profile (DJ1 adapter)',
    },
}


def normalize_profile_name(name, default='LT1'):
    key = str(name or default).strip().upper()
    if key in CSAR2_PROFILES:
        return key
    return str(default).strip().upper()


def sensor_for_profile(name, default='LT1'):
    key = normalize_profile_name(name, default=default)
    return str(CSAR2_PROFILES[key]['sensor'])
