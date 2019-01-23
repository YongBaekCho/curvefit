from hw1 import *
import unittest, numpy as np, pandas as pd, json
from compare_pandas import *

''' 
Auxiliary files needed:
    compare_pandas.py
    sun_frame.pkl, daylength_series.pkl, results_frame.pkl
    line.json, quadratic.json, cubic.json, sine.json
This one is needed by hw1.py and is therefore required:
    sunrise_sunset.csv
'''

class TestFns(unittest.TestCase):
    def test_read_frame(self):
        correct = pd.read_pickle('sun_frame.pkl')
        sf = read_frame()
        self.assertTrue((correct.index == sf.index).all())
        self.assertTrue((correct.columns == sf.columns).all())
        correct = correct.fillna('')
        sf = sf.fillna('')
        self.assertTrue((correct.values == sf.values).all())
        
    def test_get_daylength_series(self):
        correct = pd.read_pickle('daylength_series.pkl')
        sun_frame = pd.read_pickle('sun_frame.pkl')
        self.assertTrue(compare_series(correct, get_daylength_series(sun_frame), 0.00001))
   
    def test_best_fit_line(self):
        correct = json.load(open('line.json'))
        dls = pd.read_pickle('daylength_series.pkl')
        params, *stats = best_fit_line(dls)
        self.assertTrue(compare_lists(correct, list(params) + stats, 0.00001))
        self.assertFalse(np.nan in list(params) + stats)
   
    def test_best_fit_quadratic(self):
        correct = json.load(open('quadratic.json'))
        dls = pd.read_pickle('daylength_series.pkl')
        params, *stats = best_fit_parabola(dls)
        self.assertTrue(compare_lists(correct, list(params) + stats, 0.00001))
        self.assertFalse(np.nan in list(params) + stats)
   
    def test_best_fit_cubic(self):
        correct = json.load(open('cubic.json'))
        dls = pd.read_pickle('daylength_series.pkl')
        params, *stats = best_fit_cubic(dls)
        self.assertTrue(compare_lists(correct, list(params) + stats, 0.00001))
        self.assertFalse(np.nan in list(params) + stats)
        
    def test_r_squared(self):
        f = lambda x: x**2
        s = pd.Series([3.9, 1.2, -.1, 0.9, 4.23], [-2, -1, 0, 1, 2])
        self.assertTrue(abs(0.99171806096154047 - r_squared(s, f)) < 0.000001)
        s2 = pd.Series([4, 1, 0, 1, 4], [-2, -1, 0, 1, 2])
        self.assertTrue(abs(1.0 - r_squared(s2, f)) < 0.00000001)
   
    def test_best_fit_sine(self):
        correct = json.load(open('sine.json'))
        dls = pd.read_pickle('daylength_series.pkl')
        params, *stats = best_fit_sine(dls)
        self.assertTrue(compare_lists(correct, list(params) + stats, 0.00001))
        self.assertFalse(np.nan in list(params) + stats)
   
    def test_get_results_frame(self):
        correct = pd.read_pickle('results_frame.pkl')
        dls = pd.read_pickle('daylength_series.pkl')
        self.assertTrue(compare_frames(correct, get_results_frame(dls), 0.00001))
        
def main():
    test = unittest.defaultTestLoader.loadTestsFromTestCase(TestFns)
    results = unittest.TextTestRunner().run(test)
    print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 100) + ' / 100')
    dls = pd.read_pickle('daylength_series.pkl')
    rf = pd.read_pickle('results_frame.pkl')
    make_plot(dls, rf)
    
if __name__ == "__main__":
    main()