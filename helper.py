from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import sys
import locale


def unicode_input(prompt):
    return raw_input(prompt).decode(sys.stdin.encoding or
                                    locale.getpreferredencoding(True))
