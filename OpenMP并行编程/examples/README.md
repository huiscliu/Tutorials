
# OpenMP Examples
The examples are written for Linux/Unix/Mac OS, not Windows.

# How-to
## Compilation
The simplest way to configure is to run command:
```
make
```

## Run
The following command sets the number of OpenMP threads and runs the compiled codes. If the OpenMP thread
number isn't set, operating system and OpenMP running time library will set the number of threads. Usually
it's the maximal available processors.

```
export OMP_NUM_THREADS=x; ./app
```

Here **x** is the number of threads, such as 4, and **app** is the binary file.

## Clean
The following command will remove all generated files.
```
make clean
```
