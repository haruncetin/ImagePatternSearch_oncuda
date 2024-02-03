ifdef OS
    include Makefile.windows
else
	OS = $(shell uname)
	ifeq ($(OS),Linux)
		include Makefile.linux
	endif
	ifeq ($(OS),Darwin)
		include Makefile.osx
	endif
endif
