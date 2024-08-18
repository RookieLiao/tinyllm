CC = gcc

.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm

rundebug: run.c
	$(CC) -g -o run run.c -lm

.PHONY: clean
clean:
	rm -f run
