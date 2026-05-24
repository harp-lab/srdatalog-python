'''Typed Pragma subclasses for the `relation.jaccard` dialect.

Each module here defines exactly one `Pragma` subclass + its
`@pragma_handler` materialization callback. Module-import-time side
effects populate the global pragma registry; the parent
`srdatalog_jaccard` package imports each module for those side
effects.
'''
