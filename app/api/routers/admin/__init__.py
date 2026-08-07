"""Administration Portal routers (RFC-003).

Sprint 3, Phase 3.2 adds the first module -- Knowledge Management.
There is deliberately no role/auth check on these routes yet: RFC-003
scopes authentication as its own phase (User Management, not yet
scheduled), and every prior Sprint 3 phase has been told explicitly not
to build it early ("Do not build authentication"). When that phase
lands, every router here gets one added dependency
(``Depends(require_role(Role.ADMINISTRATOR))``) -- a contained addition,
not a rework, per RFC-003's Security Model.
"""
