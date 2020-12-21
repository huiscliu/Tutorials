
# Introduction to Parallel Programming Using MPI
This is a tutorial for massive parallel programming using MPI.

Many applications require huge memory and CPU performance, such as weather forecast and nuclear simulation.
Assuming we have a three-dimensional domain and each dimension is divided into 1000 intervals, then we have
10^9 elements (cells). 8 GB memory will be required for one variable, such as temperature (double precision).
Some applications have dozens or even more variables. We can see that these applications need lots of memory
and computations.

The performance of a workstation/desktop is limited by CPU performance, memory size and memory bandwidth.
It is hard for these computers to handle large-scaled applications.
Parallel computers are the only option for this applications. A parallel computer (supercomputer) has
hundreds/thousands of nodes and these nodes are connected by high-speed network. Each node is a computer.

Each node can access its own memory. When information from other nodes is needed, explicit communications have
to used to send/receive information. Let us take dot product (inner product) as an example,
```
d = <x,y>
```

In practice, each node only stores part of a vector, so each node can calculate a partial dot product only. If
a node needs the final dot product, it should collect other partial results and sum them up. Explicit
information will be sent and received.

MPI (Message Passing Interface) is designed to handle communications in parallel computing environment, which
defines a collection of API to manage messages. MPI is the most popular communication tool and is used by all
modern parallel computers.

Buy Me a Coffee & Send Tips: https://www.paypal.me/huiscliu

