from lab_multi_logi_reg import prep_report_data
from car03.multi_logistic_reg import LogisticRegression
import inspect
import unittest


class LogisticRegressionModelTest(unittest.TestCase):

    def test_model_input(self):
        sig = inspect.signature(LogisticRegression.__init__)
        # 打印类的__init__方法的参数信息
        for name, param in sig.parameters.items():
            print(f"Parameter: {name}")
            print(f"  Default value: {param.default}")
            print(f"  Annotation: {param.annotation}")
            print(f"  Kind: {param.kind}")
            print(f"  Required: {param.default == param.empty}")
            print()

    def test_shape_y_hat(self):
        plt,model,y_test,yhat,k= prep_report_data()
        self.assertEqual(yhat.shape,(len(y_test),))


if __name__ == "__main__":

    unittest.main()

