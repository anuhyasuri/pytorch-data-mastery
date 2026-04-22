import os
import unittest
import torch
from src.hf_pipeline import get_hf_loader
from src.datasets import MyImageDataset
from torchvision import transforms

#Unit testcases for HF data
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

#Unit testcases for Kaggle data
class TestKaggleDataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run before all the tests to load the data"""
        test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
        cls.root_dir = "data/archive/seg_train/seg_train"
        cls.batch_size = 4
        cls.dataset = MyImageDataset(root_dir = cls.root_dir, transform = test_transform)
        cls.loader = torch.utils.data.DataLoader(cls.dataset , batch_size = cls.batch_size)
        try:
            cls.batch  = next(iter(cls.loader))
        except Exception as e:
            cls.batch = None
            cls.error = e
    
    def test_local_loading(self):
        """Check if the local folder was found and images were loaded"""
        self.assertIsNotNone(self.batch, f"Failed to load kaggle data {getattr(self, 'error', 'Unknown error')}")

    def test_tensor_conversion(self):
        """Test if images are correct dimensions"""
        images, labels  = self.batch
        self.assertEqual(images.shape,(self.batch_size,3,224,224))
        self.assertIsInstance(images,torch.Tensor)

    def test_class_consistency(self):
        """Verify the number of folders match the number of classes"""
        num_folders = len([f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))])
        self.assertEqual(len(self.dataset.class_names),num_folders)
    


if __name__ == '__main__':
    unittest.main()