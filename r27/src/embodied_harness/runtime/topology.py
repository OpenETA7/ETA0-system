"""Model-role admission, not a semantic reviewer or an action authority."""

INDEPENDENT_ROLE_FLAGS = (
    'strong_target_judge', 'target_judge_overlap', 'strong_destination_judge',
    'strong_recovery_executor', 'physical_failure_executor',
)


def allowed_roles(mode):
    # A single-strong baseline retains the existing actor role name ``fast``.
    if mode in ('single_fast', 'single_strong'):
        return ('fast',)
    if mode in ('dual_async', 'dual_sync'):
        return ('fast', 'slow')
    raise ValueError('Unsupported mode for two-role runtime')


def validate_two_role_options(args):
    if not getattr(args, 'two_role_runtime', False):
        return
    allowed_roles(args.mode)
    conflicts = [name for name in INDEPENDENT_ROLE_FLAGS if getattr(args, name, False)]
    if conflicts:
        raise ValueError('two_role_runtime forbids independent model routes: ' + ', '.join(conflicts))
    if getattr(args, 'identity_check', True):
        raise ValueError('two_role_runtime requires --no-identity-check; legacy grounding is an independent client')


class ModelTopology:
    """Admit one client per allowed role, before calling its constructor."""

    def __init__(self, mode, enabled, store):
        self.allowed = allowed_roles(mode) if enabled else None
        self.created = []
        self.store = store

    def create(self, factory, base_url, api_key, model, role, output, **kwargs):
        if self.allowed is not None:
            if role not in self.allowed:
                raise ValueError('Model role forbidden by two-role topology: ' + role)
            if role in self.created:
                raise ValueError('Duplicate model client forbidden by two-role topology: ' + role)
        client = factory(base_url, api_key, model, role, output, **kwargs)
        self.created.append(role)
        if self.allowed is not None:
            self.store.emit('topology.client_created', {'role': role, 'model': model})
        return client

    def verify_complete(self):
        if self.allowed is not None and sorted(self.created) != sorted(self.allowed):
            raise ValueError('Incomplete two-role client topology')
