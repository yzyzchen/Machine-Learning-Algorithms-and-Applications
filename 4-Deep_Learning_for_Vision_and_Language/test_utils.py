import random
import sys
import time
import traceback
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, call
try:
    import torch
except ImportError:
    pass
import numpy

# prevent student from leaking information by calling sys.exit(answer)
VALID_ERROR_CODES = [0, 1, 2]
DEFAULT_ERROR_CODE = 1

class _TextTestResultWithCode(unittest._TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.exit_code = None

    def addError(self, test, err):
        super().addError(test, err)
        if hasattr(err[1], 'code'):
            self.exit_code = err[1].code
        else:
            self.exit_code = 1# syntax or name error

class TestRunnerWithCode(unittest.TextTestRunner):
    def _makeResult(self):
        return _TextTestResultWithCode(self.stream, self.descriptions, self.verbosity)

class TestProgramWithCode(unittest.TestProgram):
    def runTests(self):
        self.testRunner = TestRunnerWithCode(verbosity=self.verbosity)
        result = self.testRunner.run(self.test)
        if result.exit_code is None:
            if result.wasSuccessful():
                exit_code = 0
            else:
                exit_code = 1
        else:
            exit_code = result.exit_code
        if exit_code not in VALID_ERROR_CODES:
            exit_code = DEFAULT_ERROR_CODE
        sys.exit(exit_code)

class TensorTestCase(unittest.TestCase):
    @staticmethod
    def _check_shape(tensor, desired):
        assert tensor.shape == desired, (
            f"Invalid shape: {tensor.shape} != {desired}"
        )

    @staticmethod
    def _check_dtype(tensor, dtype):
        if not isinstance(dtype, (list, tuple)):
            dtype = (dtype, )
        assert tensor.dtype in dtype, (
            f"Invalid dtype: {tensor.dtype}, expected: {dtype}"
        )

    def assertTrueWith(self, val, msg=None, error_code=1):
        if not val:
            try:
                if msg is None:
                    msg = 'tensor test case assertion failed'
                raise ValueError(msg)
            except:
                traceback.print_exc()
            sys.exit(error_code)
            
    def assertTensorEqual(self, a, b, msg=None):
        self.assertEqual(a.dtype, b.dtype, "data type should be matched")
        self.assertEqual(a.shape, b.shape, "data shape should be matched")
        if msg is not None:
            output = msg
        else:
            output = "some of the tensor values do not match"
        #self.assertTrue((a == b).all(), output)
        self.assertTrueWith((a == b).all(), output, 2)

    def assertTensorClose(self, a, b, tol=1e-8, msg=None):
        self.assertEqual(a.dtype, b.dtype, "data type should be matched")
        self.assertEqual(a.shape, b.shape, "data shape should be matched")
        diff = numpy.absolute(a - b).max()
        if msg is not None:
            output = msg
        else:
            output = "the current output is far from the expected result"
        #self.assertLess(diff, tol, output)
        self.assertTrueWith(diff < tol, output, 2)

    def reset_seed(self, include_torch=False):
        random.seed(545)
        numpy.random.seed(545)
        if include_torch:
            torch.manual_seed(545)
            torch.cuda.manual_seed_all(545)

    def wrap_by_magic_mock(self, x):
        x_mock = MagicMock(wraps=x)
        x_mock.__getitem__.return_value = x
        x_mock.shape = x.shape
        x_mock.device = x.device
        x_mock.dtype = x.dtype
        # x_mock.to.return_value = x

        return x_mock

    @contextmanager
    def assertUnchanged(self, *args):
        """
        Context manager to make sure that a set of tensors is not modified.
        We can use it like this in test code:
        x1 = [create some input tensor]
        x2 = [create some input tensor]
        with self.assertUnchanged(x1, x2):
          y = f(x1, x2)
        This will make sure that the code inside the context manager does not modify
        the specified tensors x1 and x2.
        """
        # Make sure we only got tensors, and copy them
        assert all(isinstance(arg, numpy.ndarray) for arg in args)
        args_clones = [arg.copy() for arg in args]
        try:
            yield
        finally:
            for arg, arg_clone in zip(args, args_clones):
                self.assertTensorEqual(arg, arg_clone)

    @contextmanager
    def assertNotUsed(self, *fn_names):
        """
        Context manager to make sure that specific blocklisted torch functions
        or tensor methods are not used. We can use it in test code like this:
        with self.assertNotUsed('numpy.sigmoid'):
          [code to be tested]
        The context manager monkey-patches functions in the numpy namespace,
        so that calling them causes an assertion failure.
        """
        def make_new_fn(fn_name):
            def fn(*args, **kwargs):
                msg = 'Used disallowed function "%s"' % fn_name
                self.fail(msg)
            return fn

        # Apply monkeypatches specified by fn_names, and also cache a reference to
        # the original functions so we can restore them later
        old_fns = []
        for fn_name in fn_names:
            prefix, name = fn_name.split('.')
            assert prefix in ['numpy']
            if prefix == 'numpy':
                if not hasattr(numpy, name):
                    print('Skipping ', fn_name)
                    continue
                old_fns.append(getattr(numpy, name))
                setattr(numpy, name, make_new_fn(fn_name))
        try:
            yield
        finally:
            # Restore the original functions
            for fn_name, old_fn in zip(fn_names, old_fns):
                prefix, name = fn_name.split('.')
                if prefix == 'numpy' and hasattr(numpy, name):
                    setattr(numpy, name, old_fn)
