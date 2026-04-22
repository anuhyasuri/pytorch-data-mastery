import unittest
import torch
from src.hf_pipeline import get_hf_loader

class TestHFDataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run before all the tests to load the data"""
        cls.dataset = "ethz/food101"
        cls.batch_size = 4
        cls.loader = get_hf_loader(dataset = cls.dataset,batch_size=cls.batch_size)
        cls.batch = next(iter(cls.loader))

    def test_tensor_shapes(self):
        """Test if images are correct dimensions"""
        images = self.batch["pixel_values"]
        self.assertEqual(images.shape,(self.batch_size,3,224,224))
        self.assertIsInstance(images,torch.Tensor)
    
    def test_label_validity(self):
        """Test if labels are within the expected 101 classes"""
        labels = self.batch["label"]
        self.assertTrue(torch.all(labels>=0))
        self.assertTrue(torch.all(labels<101))
    
    def test_normalization(self):
        """Test if data is normalized (should not be 0-255)"""
        images = self.batch["pixel_values"]
        self.assertLess(images.mean().item(), 5.0)
        self.assertGreater(images.mean().item(), -5.0)


if __name__ == '__main__':
    unittest.main()