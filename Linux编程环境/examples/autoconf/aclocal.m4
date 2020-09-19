#
# Try to replace a command with its full pathname
#
AC_DEFUN(AC_FULLPATH, [dnl
    dnl split the command into cmd and args
    ac_cmd="`echo ${$1} | sed -e 's/[[ 	]].*$//'`"
    ac_args="`echo ${$1} | sed -e 's/^[[ 	]]*'${ac_cmd}'[[ 	]]*//'`"
    if test -z "${ac_cmd}"; then
	echo 1>&2 "***** ac_fullpath warning: empty command name in '\${$1}'."
	dnl exit 1
    else
	dnl First, try Bash builtin command 'type'
	if ac_tmp="`type -Pf ${ac_cmd}`"; then
	    if ! test -x "${ac_tmp}"; then
		ac_tmp=""
	    fi
	else
	    ac_tmp=""
	fi

	dnl Next, try 'which' command. Note: use --skip-alias --skip-functions?
	if test -z "${ac_tmp}"; then
	    if ac_tmp="`which ${ac_cmd}`"; then
		if ! test -x "${ac_tmp}"; then
		    ac_tmp=""
		fi
	    else
		ac_tmp=""
	    fi
	fi

	if test -z "${ac_tmp}"; then
	    ac_tmp=${ac_cmd}
	fi
	if ! echo "${ac_tmp}" | grep -q "^/"; then
	    ac_tmp="`pwd`/${ac_tmp}"
	fi
	$1="${ac_tmp} ${ac_args}"
	unset ac_tmp
    fi
    unset ac_cmd
    unset ac_args
])

#
# AC_ADD_FLAGS(varname, new flags)
#
AC_DEFUN(AC_ADD_FLAGS, [dnl
    for ac_i in $2; do
	if test "$ac_i" = "-I/usr/include" -o "$ac_i" = "-L/usr/lib"; then
	    continue
	fi
	if ! echo " ${$1} " | grep " $ac_i " >/dev/null; then
	    $1="${$1} $ac_i"
	    continue
	fi
    done dnl
])

#
# AC_ADD_LIBS(varname, new libs[, append_flag])
#
AC_DEFUN(AC_ADD_LIBS, [dnl
    ac_tmp=""
    for ac_i in $2; do
	# always keep '-lxxx'
	if test "$ac_i" != "`echo $ac_i | sed -e 's/^-l//'`"; then
	    ac_tmp="$ac_tmp $ac_i"
	    continue
	fi
	# always keep non-option items
	# Warning: this breaks cases like '-Ldira -lxxx -Ldirb -lxxx'
	if test "$ac_i" = "`echo $ac_i | sed -e 's/^-//'`"; then
	    ac_tmp="$ac_tmp $ac_i"
	    continue
	fi
	if ! echo " ${$1} " " $ac_tmp " | grep " $ac_i " >/dev/null; then
	    ac_tmp="$ac_tmp $ac_i"
	    continue
	fi
    done
    if test -n "$ac_tmp"; then
	if test "$3" = "append"; then
	    # append new libs
	    $1="${$1} $ac_tmp"
	else
	    # prepend new libs
	    $1="$ac_tmp ${$1}"
	fi
    fi dnl
])

#
# Change a symbolic link to its target
#
AC_DEFUN(AC_READLINK, [dnl
    $1="`echo ${$1} | sed -e 's/\/$//g'`";	dnl strip trailing '/'
    while test -h "${$1}"; do
	dnl try to get target name in turn with readlink, python, and perl
	if d=`readlink "${$1}" 2>/dev/null` || \
		dnl Python: os.path.join(os.path.dirname(path), result).
		d=`echo 'import os; print(os.readlink("'${$1}'"))' \
			| python 2>/dev/null` || \
		d=`echo 'print readlink "'${$1}'"' | perl 2>/dev/null`
	then
	    d="`echo $d | sed -e 's/\/$//g'`"	# delete trailing /
	    if test "`echo $d | cut -b1 2>/dev/null`" = "/"; then
		$1="$d"
	    else
		$1="`dirname ${$1} 2>/dev/null`/$d"
	    fi
	    continue
	else
	    dnl cannot convert, keep pathname unchanged
	    break
	fi
    done
])

#
# AC_CHECK_FORTRAN(code, ret)
#
AC_DEFUN(AC_CHECK_FORTRAN, [dnl
    AC_LINK_IFELSE([$1], $2_ok=yes, $2_ok=no)
    if test "${$2_ok}" != "yes"; then
	LIBS_forbak="${LIBS}"
	AC_ADD_LIBS(LIBS, ${FCLIBS}, append)
	AC_LINK_IFELSE([$1], $2_ok=yes, $2_ok=no)
	if test "${$2_ok}" != "yes"; then
	    LIBS="${LIBS_forbak}"
	    AC_ADD_LIBS(LIBS, ${FLIBS}, append)
	    AC_LINK_IFELSE([$1], $2_ok=yes, $2_ok=no)
	fi
	if test "${$2_ok}" != "yes"; then
	    LIBS="${LIBS_forbak}"
	fi
    fi
])

#
# AC_FIND_HEADER(var_name, dirs, dirname_pattern, header_name)
#
AC_DEFUN(AC_FIND_HEADER, [dnl
    dnl First look in the directories in $CPPFLAGS (-I options)
    if test x"${$1}" = x; then
	if test -n "${CPPFLAGS}"; then
	    for ac_d in `echo ${CPPFLAGS} | sed -e 's/-I */-I/g'`; do
		ac_f="`echo ${ac_d} | sed -e 's/^-I//g'`"
		if test "${ac_f}" == "${ac_d}"; then continue; fi
		for ac_tmp in ${ac_f}/$3/$4.h; do
		    if test -r "${ac_tmp}"; then
			$1="`dirname ${ac_tmp}`"
			break
		    fi
		done
		if test x"${$1}" != x; then break; fi
	    done
	fi
    fi
    dnl Next look in some conventional directories
    if test x"${$1}" = x; then
	for ac_d in $2 /usr/include/$3 /usr/$3/include dnl
		 /opt/include /opt/include/$3 /opt/$3/include dnl
		 /usr/local/include /usr/local/include/$3 dnl
		 /usr/local/$3/include
	do
	    for ac_f in ${ac_d}/$4.h; do
		if test -r "${ac_f}"; then
		    $1="`dirname ${ac_f}`"
		    break
		fi
	    done
	    if test x"${$1}" != x; then break; fi
	done
    fi dnl
])

#
# AC_FIND_LIB(name, dirs, dirname_pattern, libname, testprog)
#
AC_DEFUN(AC_FIND_LIB, [dnl
    if test x"${with_$1_lib}" = x; then
	ac_flag=false
	for ac_d in $2 /usr/lib* /usr/lib*/$3 /usr/$3/lib* dnl
		    /opt/lib* /opt/lib*/$3 /opt/$3/lib* dnl
		    /usr/local/lib* /usr/local/lib*/$3 /usr/local/$3/lib*
	do
	    for ac_f in ${ac_d}/lib$4.so ${ac_d}/lib$4.a; do
		if test -r "${ac_f}"; then
		    with_$1_lib="${with_$1_lib} ${ac_f}"
		fi
	    done
	done
    else
	ac_flag=true
    fi
    CPPFLAGS_findbak="${CPPFLAGS}"
    LDFLAGS_findbak="${LDFLAGS}"
    LIBS_findbak="${LIBS}"
    if test x"${with_$1_incdir}" != x; then
	AC_ADD_FLAGS(CPPFLAGS, -I$with_$1_incdir)
    fi
    for ac_lib in "" ${with_$1_lib}; do
	if ${ac_flag}; then
	    LIBS="${with_$1_lib} $LIBS_findbak"
	elif test x$ac_lib != x; then
	    dnl ac_base="`echo ${ac_lib} | sed -e 's|.*/||'`"
	    dnl ac_dir="`echo ${ac_lib} | sed -e \"s|/${ac_base}||\"`"
	    dnl if test "$ac_dir" = "$ac_base"; then ac_dir=""; fi
	    ac_base="`basename ${ac_lib}`"
	    ac_dir="`dirname ${ac_lib}`"
	    if test x$ac_dir != x; then
		if ! echo " $LDLAGS " | grep ' '$ac_dir' ' >/dev/null; then
		    LDFLAGS="$LDFLAGS_findbak"
		    AC_ADD_FLAGS(LDFLAGS, -L${ac_dir})
		fi
	    fi
	    ac_base=`echo $ac_base | dnl
			sed -e 's/^lib//' -e 's/\.a$//' -e 's/\.so$//'`
	    LIBS="-l${ac_base} ${LIBS_findbak}"
	else
	    LIBS="${LIBS_findbak}"
	fi
	AC_LINK_IFELSE([$5], enable_$1=yes, enable_$1=no)
	if ${ac_flag}; then break; fi
	if test ${enable_$1} = no; then
	    ac_lib="-l$4"
	    LIBS="-l$4 ${LIBS_findbak}"
	    AC_LINK_IFELSE([$5], enable_$1=yes, enable_$1=no)
	fi
	if test ${enable_$1} != no; then
	    with_$1_lib=${ac_lib}
	    break
	fi
    done
    if test ${enable_$1} = no; then
	LIBS="${LIBS_findbak}"
	LDFLAGS="${LDFLAGS_findbak}"
	CPPFLAGS="${CPPFLAGS_findbak}"
    fi dnl
])

