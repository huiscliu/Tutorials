
.PHONY: default all clean distclean install dep

include Makefile.inc

default:
	@(cd src; $(MAKE))

lib:
	@(cd src; $(MAKE))

all:
	@(cd src; $(MAKE))
	@(cd example; $(MAKE))

clean:
	@(cd src; $(MAKE) clean)
	@(cd example; $(MAKE) clean)
	@rm -fr config-env.log config.log config.status autom4te.cache

distclean:
	@(cd src; $(MAKE) clean)
	@(cd example; $(MAKE) clean)
	@rm -f Makefile Makefile.inc 
	@rm -fr config-env.log config.log config.status autom4te.cache
	@rm -f include/config.h Makefile Makefile.inc

install:
	@(cd src; $(MAKE) -s lib$(LIBNAME).a)
	@mkdir -p ${libdir}
	@/bin/rm -f ${libdir}/lib$(LIBNAME).a >/dev/null 2>&1 || true
	@cp src/lib$(LIBNAME).a ${libdir}/
	@chmod 0644 ${libdir}/lib$(LIBNAME).a
	@echo "installing header files to ${includedir}/"
	@mkdir -p ${includedir}
	@cp -f include/*.h  ${includedir}/
	@chmod 0644 ${includedir}/*.h
