import os
import unittest

from predict import predict_person

class TestLongHairIdentification(unittest.TestCase):

    def test_missing_image(self):
        with self.assertRaises(FileNotFoundError):
            predict_person("missing_test_image.jpg")

    def test_prediction_output(self):
        image_path = os.path.join(
            "dataset", "test", "long_hair", "test1.jpg"
        )

        if not os.path.isfile(image_path):
            self.skipTest(
                "Add dataset/test/long_hair/test1.jpg to run this test."
            )

        result = predict_person(image_path)

        self.assertIn(
            result["hair"],
            ["Long Hair", "Short Hair"],
        )
        self.assertIn(
            result["task_output"],
            ["Man", "Woman", "Male", "Female"],
        )
        self.assertGreaterEqual(result["hair_confidence"], 0)
        self.assertLessEqual(result["hair_confidence"], 100)

if __name__ == "__main__":
    unittest.main()