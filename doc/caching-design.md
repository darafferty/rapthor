# Changes to the Proxy interface to enable caching strategies

The interface of the Proxy has been changed to allow proxies to implement caching strategies.
In the previous interface in a series of subsequent calls to the `gridding(...)` method
each call was independent from the other. Independent in the sense that for each call all parameters describing the job were passed as arguments. The same applied to a series of `degridding(...)` calls.

In practice often a series of calls is made using the same parameters, while only the data is different. Proxies can implement more efficient strategies when it is known beforehand that a series of similar calls is about to be executed. Examples of these are (and currently the only ones implemented)

1. The CUDA proxies keeping the grid in device memory until the calls have finished.
2. The CPU and Hybrid proxies using W-tiling, a caching strategy keeping only a small, currently active, portion of a huge virtual W-stack in memory.

The interface has been changed to support these strategies. The old gridding and degridding methods, accepting all parameters, have been removed. The old forms could have been provided as convenience layer around the new methods, but the performance would be a lot lower, because they would effectively disable caching.

Caching strategies may require additional information to be put in the Plan, depending on what strategy the Proxy uses. How the Plan needs to be made is known only to the Proxy. Therefore plans can now only be created by a call to the `make_plan(...)` method of the Proxy.

Caching strategies may postpone updates to the grid. To get the final grid after a (series of) `gridding(...)` call(s), the `get_final_grid()` method need to be called.

### Gridding
In the old interface there were multiple varieties of the `gridding(...)` method. The shortest form was:
```
proxy.gridding(w_step, shift, float cell_size, kernel_size, subgrid_size, frequencies, visibilities, uvw, baselines, grid, aterms, aterm_offsets, taper);
```

In the new interface the only way is the following series of calls, in this precise order:
```
proxy.set_grid(grid);
proxy.init_cache(subgrid_size, cell_size, w_step, shift);
std::unique_ptr<Plan> plan = proxy.make_plan(kernel_size, frequencies, uvw, baselines, aterm_offsets, options);
proxy.gridding(*plan, frequencies, visibilities, uvw, baselines, aterms, aterm_offsets, taper);
proxy.get_grid();
```
where the `make_plan(...); gridding(...);` calls can be repeated as many times as there blocks of data to be gridded.

### Degridding
Degridding is similar to gridding, except that there is no final call to the `get_final_grid()` method.
The equivalent of the old
```
proxy.degridding(w_step, shift, float cell_size, kernel_size, subgrid_size, frequencies, visibilities, uvw, baselines, grid, aterms, aterm_offsets, taper);
```
is
```
proxy.set_grid(grid);
proxy.init_cache(subgrid_size, cell_size, w_step, shift);
std::unique_ptr<Plan> plan = proxy.make_plan(kernel_size, frequencies, uvw, baselines, aterm_offsets, options);
proxy.degridding(*plan, frequencies, visibilities, uvw, baselines, aterms, aterm_offsets, taper);
```
where the `make_plan(...); degridding(...);` calls can be repeated as many times as there blocks of data to be degridded.
