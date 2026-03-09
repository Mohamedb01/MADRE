try:
    import pandas
    print("pandas: installed")
    import numpy
    print("numpy: installed")
    import matplotlib
    print("matplotlib: installed")
    import sklearn
    print("sklearn: installed")
    import tensorflow
    print("tensorflow: installed")
    import imblearn
    print("imblearn: installed")
    import category_encoders
    print("category_encoders: installed")
except ImportError as e:
    print(f"Missing library: {e}")
