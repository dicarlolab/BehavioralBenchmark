import sys

import hvm_2way_consistency as h

feature_name = sys.argv[1]
consistency_type = sys.argv[2]
h.store_consistency(feature_name, consistency_type)
