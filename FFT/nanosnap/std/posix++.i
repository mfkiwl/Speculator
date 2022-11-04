%module posix
%{
#include "posix++.h"
using namespace posix;
%}

%include "posix++/descriptor.h"
%include "posix++/directory.h"
%include "posix++/error.h"
%include "posix++/feature.h"
%include "posix++/file.h"
%include "posix++/group.h"
%include "posix++/local_socket.h"
%include "posix++/mapped_file.h"
%include "posix++/memory_mapping.h"
%include "posix++/message_queue.h"
%include "posix++/mode.h"
%include "posix++/module.h"
%include "posix++/named_pipe.h"
%include "posix++/pathname.h"
%include "posix++/process.h"
%include "posix++/process_group.h"
%include "posix++/semaphore.h"
%include "posix++/socket.h"
%include "posix++/stdio.h"
%include "posix++/sysv_segment.h"
%include "posix++/thread.h"
%include "posix++/user.h"
%include "posix++/version.h"

