
Should we keep supporting usage without caching?
-> for now yes!



set_grid()
init_cache()

gridding() or degridding() calls

flush_cache()

What should flush_cache do after degridding() calls? No-op, or incorrect behaviour?
-> No-op
