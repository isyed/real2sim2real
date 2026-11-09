# Sharing a bounded local GPU clock session

`real2sim2real.gpu_sharing` coordinates completion when explicitly authorized
jobs use one already active clock lease. It does not change clocks, grant GPU
permission, replace thermal supervision or allocate GPU memory. The privileged
clock helper keeps its original maximum lifetime and hardware-restoration duty.

Register a named borrower before the owner can finish. Supply a bounded number
of seconds; the claim is capped by the existing lease deadline minus a safety
reserve. Each workload must independently check the real active lease and apply
its temperature, clock, memory and execution limits. If combined memory or
thermal use is excessive, schedule sequentially rather than assuming the claim
is a resource scheduler.

The owner calls `owner_finished(session)` after its final GPU work. Borrowers
call `borrower_finished(session,name)` after theirs. Neither completion closes
the session while the other still needs it. A bounded `watch(session)` can
release a completed owner's session after the last borrower finishes or expires.
Older owner-completion records cannot close a later lease. CPU rendering may
continue after GPU completion. Registering a borrower does not approve its
scientific result or allow it to bypass failed input-quality gates.

If an owner process is already running, adopt cooperative completion only at
an idle supervisor boundary. Preserve running CPU/GPU work and all numeric
inputs. The registration and coordination must be explicit in the job record;
another process must not remove or race the owner's completion marker.
