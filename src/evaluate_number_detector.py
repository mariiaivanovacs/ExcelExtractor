#!/usr/bin/env python3
"""
Reproducible evaluation script for number_detector_cnn.keras
Evaluates binary classification (NUMBER vs OTHER) with confusion matrix and detailed metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from pathlib import Path
import json

class NumberDetectorEvaluator:
    def __init__(self, model_path="mnt/outputs/number_detector_cnn.keras"):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the trained number detector model
        """
        self.model_path = model_path
        self.model = None
        self.img_size = (32, 32)  # Adjust based on your model's input size
        self.class_names = ['OTHER', 'NUMBER']
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        self.model = load_model(self.model_path)
        print("✅ Model loaded successfully")
        
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to model input size
        img = cv2.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        img = img.reshape(1, self.img_size[0], self.img_size[1], 1)
        
        return img
        
    def predict_single(self, image_path):
        """
        Make prediction for a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            img_tensor = self.preprocess_image(image_path)
            predictions = self.model.predict(img_tensor, verbose=0)
            
            # Handle both binary classification outputs
            if predictions.shape[1] == 2:
                # Two-class output [OTHER, NUMBER]
                predicted_class = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]))
                number_probability = float(predictions[0][1])
            else:
                # Single output (sigmoid)
                number_probability = float(predictions[0][0])
                predicted_class = 1 if number_probability > 0.5 else 0
                confidence = number_probability if predicted_class == 1 else (1 - number_probability)
            
            return {
                'filename': os.path.basename(image_path),
                'predicted_class': int(predicted_class),
                'predicted_label': self.class_names[predicted_class],
                'confidence': confidence,
                'number_probability': number_probability,
                'other_probability': 1 - number_probability
            }
        except Exception as e:
            return {
                'filename': os.path.basename(image_path),
                'error': str(e)
            }
    
    def load_ground_truth(self, ground_truth_file):
        """
        Load ground truth labels from file
        
        Args:
            ground_truth_file: Path to CSV file with columns 'filename' and 'true_label'
                              where true_label is either 'NUMBER' or 'OTHER'
            
        Returns:
            Dictionary mapping filename to true class (0=OTHER, 1=NUMBER)
        """
        if not os.path.exists(ground_truth_file):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        df = pd.read_csv(ground_truth_file)
        required_columns = ['filename', 'true_label']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Ground truth file must contain columns: {required_columns}")
        
        ground_truth = {}
        for _, row in df.iterrows():
            label = row['true_label'].upper()
            if label not in ['NUMBER', 'OTHER']:
                raise ValueError(f"Invalid label '{label}' for {row['filename']}. Must be 'NUMBER' or 'OTHER'")
            
            ground_truth[row['filename']] = 1 if label == 'NUMBER' else 0
        
        print(f"✅ Loaded ground truth for {len(ground_truth)} images")
        return ground_truth
    
    def evaluate_on_test_set(self, test_dir, ground_truth_file, output_dir="results/number_detector_evaluation"):
        """
        Evaluate model on test set with ground truth
        
        Args:
            test_dir: Directory containing test images
            ground_truth_file: CSV file with ground truth labels
            output_dir: Directory to save evaluation results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load ground truth
        ground_truth = self.load_ground_truth(ground_truth_file)
        
        # Get test images
        test_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            test_images.extend(Path(test_dir).glob(ext))
        
        print(f"Found {len(test_images)} test images in {test_dir}")
        
        # Make predictions
        results = []
        y_true = []
        y_pred = []
        y_scores = []
        
        print("Making predictions...")
        for img_path in test_images:
            filename = img_path.name
            
            if filename not in ground_truth:
                print(f"⚠️  No ground truth for {filename}, skipping")
                continue
            
            result = self.predict_single(str(img_path))
            
            if 'error' in result:
                print(f"❌ Error processing {filename}: {result['error']}")
                continue
            
            true_class = ground_truth[filename]
            result['true_class'] = true_class
            result['true_label'] = self.class_names[true_class]
            result['correct'] = (result['predicted_class'] == true_class)
            
            results.append(result)
            y_true.append(true_class)
            y_pred.append(result['predicted_class'])
            y_scores.append(result['number_probability'])
            
            # Print result
            status = "✅" if result['correct'] else "❌"
            print(f"{status} {filename}: True={result['true_label']}, "
                  f"Pred={result['predicted_label']} (conf={result['confidence']:.3f})")
        
        if not results:
            print("❌ No valid predictions made!")
            return
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n📊 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_file = os.path.join(output_dir, "detailed_results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"💾 Detailed results saved to: {results_file}")
        
        # Generate confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, output_dir)
        
        # Generate ROC curve
        self.plot_roc_curve(y_true, y_scores, output_dir)
        
        # Generate classification report
        self.generate_classification_report(y_true, y_pred, output_dir)
        
        # Generate summary statistics
        self.generate_summary_stats(results, output_dir)
        
        print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")
        
    def plot_confusion_matrix(self, y_true, y_pred, output_dir):
        """Generate and save confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Number Detection')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        cm_file = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Confusion matrix saved to: {cm_file}")
        
    def plot_roc_curve(self, y_true, y_scores, output_dir):
        """Generate and save ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Number Detection')
        plt.legend(loc="lower right")
        
        roc_file = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 ROC curve saved to: {roc_file}")
        
    def generate_classification_report(self, y_true, y_pred, output_dir):
        """Generate and save classification report"""
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        # Save as JSON
        report_file = os.path.join(output_dir, "classification_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as text
        report_text = classification_report(y_true, y_pred, target_names=self.class_names)
        report_text_file = os.path.join(output_dir, "classification_report.txt")
        with open(report_text_file, 'w') as f:
            f.write(report_text)
        
        print(f"📋 Classification report saved to: {report_file}")
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(report_text)
        
    def generate_summary_stats(self, results, output_dir):
        """Generate summary statistics"""
        total_predictions = len(results)
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / total_predictions
        
        # Separate by true class
        number_results = [r for r in results if r['true_class'] == 1]
        other_results = [r for r in results if r['true_class'] == 0]
        
        summary = {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'class_distribution': {
                'number_samples': len(number_results),
                'other_samples': len(other_results)
            },
            'confidence_stats': {
                'overall_mean': np.mean([r['confidence'] for r in results]),
                'number_mean': np.mean([r['confidence'] for r in number_results]) if number_results else 0,
                'other_mean': np.mean([r['confidence'] for r in other_results]) if other_results else 0
            }
        }
        
        summary_file = os.path.join(output_dir, "summary_stats.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary statistics saved to: {summary_file}")

def main():
    """Main evaluation function"""
    print("🔍 Number Detector Evaluation")
    print("="*50)
    
    # Initialize evaluator
    evaluator = NumberDetectorEvaluator()
    evaluator.load_model()
    
    # Check for ground truth file
    ground_truth_file = "data/ground_truth/number_detector_test_labels.csv"
    
    if not os.path.exists(ground_truth_file):
        print(f"\n⚠️  Ground truth file not found: {ground_truth_file}")
        print("\nPlease create a CSV file with the following format:")
        print("filename,true_label")
        print("cell_r19_c17_blob_1_word_4_char_02.png,NUMBER")
        print("cell_r19_c18_blob_1_word_2_char_00.png,OTHER")
        print("...")
        print("\nExample test images found:")
        
        # Show available test images
        test_dirs = ["tests", "tests_2"]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = list(Path(test_dir).glob("*.png"))[:5]
                print(f"\n{test_dir}/:")
                for img in images:
                    print(f"  {img.name}")
        
        return
    
    # Run evaluation on both test directories
    test_dirs = ["tests", "tests_2"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\n🔍 Evaluating on {test_dir}/")
            output_dir = f"results/number_detector_evaluation_{test_dir}"
            evaluator.evaluate_on_test_set(test_dir, ground_truth_file, output_dir)

if __name__ == "__main__":
    main()
