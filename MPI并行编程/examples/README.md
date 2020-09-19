
# MPI Examples
The examples are written for Linux/Unix/Mac OS, not Windows.

# How-to
## Compilation
The simplest way to configure is to run command:
```
make
```

## Run
The following command sets the number of MPI processes and runs the compiled codes.

```
mpirun -np x ./app
```

Here **x** is the number of MPI processes, such as 4, and **app** is the binary file.

## Clean
The following command will remove all generated files.
```
make clean
```
