"""
Seam Region vs Center Region mIoU Analysis
验证假设: 360°全景图接缝处(x=0/x=2048)的mIoU是否低于中心区域

Author: Research Team
Date: 2026-08-28
"""

import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from collections import defaultdict

class SeamRegionAnalyzer:
    def __init__(self, pred_dir, gt_dir, num_classes=13):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.num_classes = num_classes
        self.ignore_label = 255
        
    def compute_iou_per_class(self, pred_region, gt_region):
        """计算单个区域的per-class IoU"""
        ious = []
        for cls in range(self.num_classes):
            pred_mask = (pred_region == cls)
            gt_mask = (gt_region == cls)
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            if union == 0:
                ious.append(float('nan'))  # 该类别不存在
            else:
                ious.append(intersection / union * 100.0)
        
        # mIoU: 忽略nan值
        valid_ious = [iou for iou in ious if not np.isnan(iou)]
        miou = np.mean(valid_ious) if valid_ious else 0.0
        
        return miou, ious
    
    def extract_region(self, img_array, x_range):
        """提取指定x范围的区域"""
        x_start, x_end = x_range
        return img_array[:, x_start:x_end]
    
    def analyze_single_image(self, pred_path, gt_path, regions):
        """
        分析单张图像的多个区域
        
        Args:
            regions: dict, e.g. {
                'seam_left': (0, 100),
                'seam_right': (1948, 2048),
                'center': (900, 1148)
            }
        
        Returns:
            results: dict of {region_name: {'miou': float, 'per_class_iou': list}}
        """
        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))
        
        # 过滤ignore label
        valid_mask = (gt != self.ignore_label)
        
        results = {}
        for region_name, x_range in regions.items():
            pred_region = self.extract_region(pred, x_range)[valid_mask[:, x_range[0]:x_range[1]]]
            gt_region = self.extract_region(gt, x_range)[valid_mask[:, x_range[0]:x_range[1]]]
            
            miou, per_class_iou = self.compute_iou_per_class(pred_region, gt_region)
            results[region_name] = {
                'miou': miou,
                'per_class_iou': per_class_iou,
                'pixel_count': gt_region.size
            }
        
        return results
    
    def analyze_all(self, seam_width=100):
        """
        分析所有图像
        
        Args:
            seam_width: 接缝区域宽度(像素)
        """
        # 定义三个区域
        regions = {
            'seam_left': (0, seam_width),
            'seam_right': (2048 - seam_width, 2048),
            'center': (1024 - seam_width//2, 1024 + seam_width//2)  # 中心区域同等宽度
        }
        
        # 收集所有样本文件
        pred_files = sorted([f for f in os.listdir(self.pred_dir) if f.endswith('.png')])
        
        # 累积统计
        accumulated = {
            region: {
                'miou_list': [],
                'per_class_iou_sum': np.zeros(self.num_classes),
                'per_class_count': np.zeros(self.num_classes)
            }
            for region in regions.keys()
        }
        
        print(f"Analyzing {len(pred_files)} images...")
        print(f"Region definitions:")
        for name, (x1, x2) in regions.items():
            print(f"  {name:12s}: x ∈ [{x1:4d}, {x2:4d}), width={x2-x1}px")
        print()
        
        for pred_file in tqdm(pred_files, desc="Processing"):
            # 构造对应的GT文件名
            # pred: XXX_pred_argmax.png → gt: XXX_labelTrainIds.png
            sample_id = pred_file.replace('_pred_argmax.png', '')
            gt_file = f"{sample_id}_labelTrainIds.png"
            
            pred_path = os.path.join(self.pred_dir, pred_file)
            gt_path = os.path.join(self.gt_dir, gt_file)
            
            if not os.path.exists(gt_path):
                print(f"Warning: GT not found for {sample_id}, skipping")
                continue
            
            # 分析这张图
            results = self.analyze_single_image(pred_path, gt_path, regions)
            
            # 累积
            for region_name, res in results.items():
                accumulated[region_name]['miou_list'].append(res['miou'])
                
                # per-class累积
                for cls, iou in enumerate(res['per_class_iou']):
                    if not np.isnan(iou):
                        accumulated[region_name]['per_class_iou_sum'][cls] += iou
                        accumulated[region_name]['per_class_count'][cls] += 1
        
        # 计算最终统计
        final_results = {}
        for region_name, acc in accumulated.items():
            miou_mean = np.mean(acc['miou_list'])
            miou_std = np.std(acc['miou_list'])
            
            # per-class平均
            per_class_iou_avg = []
            for cls in range(self.num_classes):
                if acc['per_class_count'][cls] > 0:
                    avg = acc['per_class_iou_sum'][cls] / acc['per_class_count'][cls]
                else:
                    avg = float('nan')
                per_class_iou_avg.append(avg)
            
            final_results[region_name] = {
                'miou_mean': float(miou_mean),
                'miou_std': float(miou_std),
                'per_class_iou': [float(x) if not np.isnan(x) else None for x in per_class_iou_avg],
                'sample_count': len(acc['miou_list'])
            }
        
        return final_results, regions

    def print_results(self, results, regions):
        """打印结果"""
        print("\n" + "="*80)
        print("SEAM REGION vs CENTER REGION mIoU ANALYSIS")
        print("="*80)
        
        # 按区域打印
        for region_name in ['seam_left', 'seam_right', 'center']:
            res = results[region_name]
            x1, x2 = regions[region_name]
            print(f"\n{region_name.upper():12s} (x ∈ [{x1}, {x2})):")
            print(f"  mIoU: {res['miou_mean']:.2f} ± {res['miou_std']:.2f}")
            print(f"  Samples: {res['sample_count']}")
        
        # 计算接缝vs中心的gap
        seam_combined_miou = (results['seam_left']['miou_mean'] + 
                              results['seam_right']['miou_mean']) / 2
        center_miou = results['center']['miou_mean']
        gap = seam_combined_miou - center_miou
        
        print("\n" + "-"*80)
        print("KEY FINDINGS:")
        print(f"  Seam (avg of left+right): {seam_combined_miou:.2f} mIoU")
        print(f"  Center:                   {center_miou:.2f} mIoU")
        print(f"  Gap (seam - center):      {gap:+.2f} mIoU")
        print("-"*80)
        
        if gap < -1.0:
            print("\n✅ HYPOTHESIS CONFIRMED: Seam region has significantly lower mIoU!")
            print("   → Strong motivation for Seamless Circular Memory Bank")
        elif gap < -0.3:
            print("\n⚠️  HYPOTHESIS PARTIALLY SUPPORTED: Seam shows slight degradation")
            print("   → Moderate motivation, may need wider seam region or more samples")
        else:
            print("\n❌ HYPOTHESIS NOT SUPPORTED: No significant seam degradation detected")
            print("   → May need to reconsider SCM priority or check data quality")
        
        print("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Seam region mIoU analysis')
    parser.add_argument('--pred_dir', type=str, 
                        default='experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain',
                        help='Directory containing prediction masks')
    parser.add_argument('--gt_dir', type=str,
                        default='data/DensePASS/gtFine/val',
                        help='Directory containing ground truth masks')
    parser.add_argument('--seam_width', type=int, default=100,
                        help='Width of seam region (pixels)')
    parser.add_argument('--output', type=str,
                        default='experiments/tecma/diagnostics/seam_vs_center_miou.json',
                        help='Output JSON file')
    parser.add_argument('--num_classes', type=int, default=13,
                        help='Number of classes')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = SeamRegionAnalyzer(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        num_classes=args.num_classes
    )
    
    # 运行分析
    results, regions = analyzer.analyze_all(seam_width=args.seam_width)
    
    # 打印结果
    analyzer.print_results(results, regions)
    
    # 保存JSON
    output_data = {
        'regions': {name: list(x_range) for name, x_range in regions.items()},
        'results': results,
        'metadata': {
            'pred_dir': args.pred_dir,
            'gt_dir': args.gt_dir,
            'seam_width': args.seam_width,
            'num_classes': args.num_classes
        }
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {args.output}")
