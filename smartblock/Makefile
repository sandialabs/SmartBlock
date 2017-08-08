ADIOS_DIR=/net/hp100/achampsa/adios11Install

ADIOS_INC := $(shell ${ADIOS_DIR}/bin/adios_config -c)
ADIOS_FLIB := $(shell ${ADIOS_DIR}/bin/adios_config -l)
CFLAGS = -I. -fPIC -g $(ADIOS_INC)
CPPFLAGS = -DENABLE_MONITOR
LIB = -lm $(ADIOS_FLIB) #-lenet

## TITAN ##
#CC=cc
#CXX=CC

## RHEA, Falcon ##
CC=mpicc
CXX=mpic++

OUTROOT := install
HDEPS=lib_mem_monitor.h
LDEPS=${OUTROOT}/lib/lib_mem_monitor.a
LINK=-L${OUTROOT}/lib -l_mem_monitor

bins = select magnitude histogram dimreduce dimreduce2 #lammps_analysis

all: clean dirs lib $(bins)

dirs: 
	mkdir -p ${OUTROOT}/bin ${OUTROOT}/lib \
		${OUTROOT}/include ${OUTROOT}/build
	cp lib_mem_monitor.h ${OUTROOT}/include/lib_mem_monitor.h

lib: lib_mem_monitor.c $(HDEPS)
	@echo "CC	$@"
	@$(CC) -c $(CFLAGS) $< -o ${OUTROOT}/build/lib_mem_monitor.o
	@ar r ${OUTROOT}/lib/lib_mem_monitor.a ${OUTROOT}/build/lib_mem_monitor.o

$(bins): %: %.o 
	@echo "CC	$@"
	@$(CXX) -o ${OUTROOT}/bin/$@ ${OUTROOT}/build/$< $(LIB) $(LINK)

%.o: %.cpp $(HDEPS)
	@echo "CC	$@"
	@$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o ${OUTROOT}/build/$@

%.o: %.c $(HDEPS)
	@echo "CC	$@"
	@$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o ${OUTROOT}/build/$@

clean:
	rm -rf install *.txt core*
