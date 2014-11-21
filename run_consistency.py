import hvm_2way_consistency as h
import sys
feature_name = sys.argv[1]
consistency_type = sys.argv[2]
h.store_consistency(feature_name+'_results', consistency_type)
