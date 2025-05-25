# evaluation_metrics.py (补充完整实现)  
import cv2  
import numpy as np  
from skimage.metrics import structural_similarity as ssim  
from scipy.spatial.distance import cosine  
import torch  
from torchvision import transforms  
from PIL import Image  
import cairosvg  
import io  
from pathlib import Path  

class ChairGenerationEvaluator:  
    """椅子生成评估器"""  
    
    def __init__(self):  
        self.transform = transforms.Compose([  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   
                               std=[0.229, 0.224, 0.225])  
        ])  
    
    def evaluate_visual_similarity(self, generated_svg, reference_svg):  
        """评估视觉相似性"""  
        try:  
            # 将SVG转换为图像  
            gen_img = self._svg_to_image(generated_svg)  
            ref_img = self._svg_to_image(reference_svg)  
            
            if gen_img is None or ref_img is None:  
                return {'error': 'Failed to convert SVG to image'}  
            
            # 确保图像尺寸一致  
            height, width = min(gen_img.shape[0], ref_img.shape[0]), min(gen_img.shape[1], ref_img.shape[1])  
            gen_img = cv2.resize(gen_img, (width, height))  
            ref_img = cv2.resize(ref_img, (width, height))  
            
            # 转换为灰度图像计算SSIM  
            gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY) if len(gen_img.shape) == 3 else gen_img  
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY) if len(ref_img.shape) == 3 else ref_img  
            
            # 计算SSIM  
            ssim_score = ssim(gen_gray, ref_gray)  
            
            # 计算MSE  
            mse = np.mean((gen_img.astype(np.float64) - ref_img.astype(np.float64)) ** 2)  
            
            # 计算PSNR  
            if mse > 0:  
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))  
            else:  
                psnr = float('inf')  
            
            return {  
                'ssim': float(ssim_score),  
                'mse': float(mse),  
                'psnr': float(psnr)  
            }  
        except Exception as e:  
            return {'error': f'Visual similarity evaluation failed: {str(e)}'}  
    
    def evaluate_style_consistency(self, generated_views, style_features):  
        """评估风格一致性"""  
        try:  
            # 提取每个视图的特征  
            features = []  
            for view_path in generated_views:  
                if Path(view_path).exists():  
                    img = self._svg_to_image(view_path)  
                    if img is not None:  
                        feature = self._extract_style_features(img)  
                        features.append(feature)  
            
            if len(features) < 2:  
                return 0.0  
            
            # 计算视图间的一致性  
            consistency_scores = []  
            for i in range(len(features)):  
                for j in range(i+1, len(features)):  
                    # 使用余弦相似度计算特征相似性  
                    similarity = 1 - cosine(features[i].flatten(), features[j].flatten())  
                    consistency_scores.append(similarity)  
            
            return float(np.mean(consistency_scores)) if consistency_scores else 0.0  
        except Exception as e:  
            print(f"Style consistency evaluation failed: {str(e)}")  
            return 0.0  
    
    def evaluate_geometric_accuracy(self, generated_views):  
        """评估几何准确性"""  
        try:  
            views = {}  
            for view_name, view_path in generated_views.items():  
                if Path(view_path).exists():  
                    img = self._svg_to_image(view_path)  
                    if img is not None:  
                        views[view_name] = img  
            
            if len(views) < 3:  
                return 0.0  
            
            # 提取轮廓和关键点  
            contours = {}  
            for view_name, img in views.items():  
                contours[view_name] = self._extract_contours(img)  
            
            # 计算几何一致性分数  
            geometric_score = self._compute_geometric_consistency(contours)  
            
            return float(geometric_score)  
        except Exception as e:  
            print(f"Geometric accuracy evaluation failed: {str(e)}")  
            return 0.0  
    
    def comprehensive_evaluation(self, generated_views, reference_views, style_description):  
        """综合评估"""  
        results = {}  
        
        # 视觉相似性评估  
        for view in ['front', 'side', 'top']:  
            if (view in generated_views and view in reference_views and   
                Path(generated_views[view]).exists() and Path(reference_views[view]).exists()):  
                visual_sim = self.evaluate_visual_similarity(  
                    generated_views[view], reference_views[view]  
                )  
                results[f'{view}_visual_similarity'] = visual_sim  
        
        # 风格一致性评估  
        if generated_views:  
            style_consistency = self.evaluate_style_consistency(  
                list(generated_views.values()), style_description  
            )  
            results['style_consistency'] = style_consistency  
        
        # 几何准确性评估  
        if generated_views:  
            geometric_accuracy = self.evaluate_geometric_accuracy(generated_views)  
            results['geometric_accuracy'] = geometric_accuracy  
        
        # 综合分数  
        results['overall_score'] = self._compute_overall_score(results)  
        
        return results  
    
    def _svg_to_image(self, svg_path):  
        """将SVG转换为图像"""  
        try:  
            svg_path = Path(svg_path)  
            if not svg_path.exists():  
                print(f"SVG file not found: {svg_path}")  
                return None  
            
            # 读取SVG文件  
            with open(svg_path, 'r', encoding='utf-8') as f:  
                svg_content = f.read()  
            
            # 使用cairosvg将SVG转换为PNG  
            png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))  
            
            # 将PNG数据转换为PIL图像  
            img = Image.open(io.BytesIO(png_data))  
            
            # 转换为RGB模式  
            if img.mode != 'RGB':  
                img = img.convert('RGB')  
            
            # 转换为numpy数组  
            img_array = np.array(img)  
            
            return img_array  
        except Exception as e:  
            print(f"Error converting SVG to image: {str(e)}")  
            return None  
    
    def _extract_style_features(self, image):  
        """提取图像风格特征"""  
        try:  
            # 转换为灰度图  
            if len(image.shape) == 3:  
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
            else:  
                gray = image  
            
            # 提取多种特征  
            features = []  
            
            # 1. 纹理特征 (LBP)  
            lbp_features = self._compute_lbp_features(gray)  
            features.extend(lbp_features)  
            
            # 2. 边缘特征  
            edge_features = self._compute_edge_features(gray)  
            features.extend(edge_features)  
            
            # 3. 形状特征  
            shape_features = self._compute_shape_features(gray)  
            features.extend(shape_features)  
            
            return np.array(features)  
        except Exception as e:  
            print(f"Error extracting style features: {str(e)}")  
            return np.zeros(100)  # 返回零向量作为默认值  
    
    def _compute_lbp_features(self, gray_image):  
        """计算LBP纹理特征"""  
        try:  
            from skimage.feature import local_binary_pattern  
            
            # 计算LBP  
            radius = 3  
            n_points = 8 * radius  
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')  
            
            # 计算直方图  
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))  
            
            # 归一化  
            hist = hist.astype(float)  
            hist /= (hist.sum() + 1e-8)  
            
            return hist.tolist()  
        except:  
            return [0.0] * 26  # 默认返回26维零向量  
    
    def _compute_edge_features(self, gray_image):  
        """计算边缘特征"""  
        try:  
            # Canny边缘检测  
            edges = cv2.Canny(gray_image, 50, 150)  
            
            # 计算边缘密度  
            edge_density = np.sum(edges > 0) / edges.size  
            
            # 计算边缘方向直方图  
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  
            
            angles = np.arctan2(sobel_y, sobel_x)  
            angle_hist, _ = np.histogram(angles.ravel(), bins=8, range=(-np.pi, np.pi))  
            angle_hist = angle_hist.astype(float)  
            angle_hist /= (angle_hist.sum() + 1e-8)  
            
            features = [edge_density] + angle_hist.tolist()  
            return features  
        except:  
            return [0.0] * 9  # 默认返回9维零向量  
    
    def _compute_shape_features(self, gray_image):  
        """计算形状特征"""  
        try:  
            # 二值化  
            _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)  
            
            # 查找轮廓  
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            
            if not contours:  
                return [0.0] * 10  
            
            # 找到最大轮廓  
            largest_contour = max(contours, key=cv2.contourArea)  
            
            # 计算形状特征  
            area = cv2.contourArea(largest_contour)  
            perimeter = cv2.arcLength(largest_contour, True)  
            
            # 紧凑性  
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0  
            
            # 长宽比  
            x, y, w, h = cv2.boundingRect(largest_contour)  
            aspect_ratio = w / h if h > 0 else 0  
            
            # 填充率  
            rect_area = w * h  
            extent = area / rect_area if rect_area > 0 else 0  
            
            # 凸包相关特征  
            hull = cv2.convexHull(largest_contour)  
            hull_area = cv2.contourArea(hull)  
            solidity = area / hull_area if hull_area > 0 else 0  
            
            # Hu矩特征  
            moments = cv2.moments(largest_contour)  
            hu_moments = cv2.HuMoments(moments).flatten()  
            
            features = [compactness, aspect_ratio, extent, solidity] + hu_moments[:6].tolist()  
            return features  
        except:  
            return [0.0] * 10  # 默认返回10维零向量  
    
    def _extract_contours(self, image):  
        """提取图像轮廓"""  
        try:  
            if len(image.shape) == 3:  
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
            else:  
                gray = image  
            
            # 二值化  
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  
            
            # 查找轮廓  
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            
            return contours  
        except Exception as e:  
            print(f"Error extracting contours: {str(e)}")  
            return []  
    
    def _compute_geometric_consistency(self, contours_dict):  
        """计算几何一致性"""  
        try:  
            if len(contours_dict) < 3:  
                return 0.0  
            
            # 为每个视图计算基本几何特征  
            view_features = {}  
            
            for view_name, contours in contours_dict.items():  
                if not contours:  
                    view_features[view_name] = {'area': 0, 'perimeter': 0, 'bbox_ratio': 0}  
                    continue  
                
                # 找到最大轮廓  
                largest_contour = max(contours, key=cv2.contourArea)  
                
                # 计算面积和周长  
                area = cv2.contourArea(largest_contour)  
                perimeter = cv2.arcLength(largest_contour, True)  
                
                # 计算边界框长宽比  
                x, y, w, h = cv2.boundingRect(largest_contour)  
                bbox_ratio = w / h if h > 0 else 0  
                
                view_features[view_name] = {  
                    'area': area,  
                    'perimeter': perimeter,  
                    'bbox_ratio': bbox_ratio  
                }  
            
            # 计算视图间的几何一致性  
            consistency_scores = []  
            
            # 检查面积比例的一致性（前视图和侧视图的面积应该相近）  
            if 'front' in view_features and 'side' in view_features:  
                front_area = view_features['front']['area']  
                side_area = view_features['side']['area']  
                if front_area > 0 and side_area > 0:  
                    area_ratio = min(front_area, side_area) / max(front_area, side_area)  
                    consistency_scores.append(area_ratio)  
            
            # 检查长宽比的合理性  
            if 'top' in view_features:  
                top_ratio = view_features['top']['bbox_ratio']  
                if 0.5 <= top_ratio <= 2.0:  # 合理的长宽比范围  
                    consistency_scores.append(1.0)  
                else:  
                    consistency_scores.append(0.5)  
            
            return np.mean(consistency_scores) if consistency_scores else 0.0  
        except Exception as e:  
            print(f"Error computing geometric consistency: {str(e)}")  
            return 0.0  
    
    def _compute_overall_score(self, results):  
        """计算综合分数"""  
        try:  
            scores = []  
            weights = []  
            
            # 视觉相似性分数 (权重: 0.4)  
            visual_scores = []  
            for view in ['front', 'side', 'top']:  
                key = f'{view}_visual_similarity'  
                if key in results and isinstance(results[key], dict) and 'ssim' in results[key]:  
                    visual_scores.append(results[key]['ssim'])  
            
            if visual_scores:  
                scores.append(np.mean(visual_scores))  
                weights.append(0.4)  
            
            # 风格一致性分数 (权重: 0.3)  
            if 'style_consistency' in results and isinstance(results['style_consistency'], (int, float)):  
                scores.append(results['style_consistency'])  
                weights.append(0.3)  
            
            # 几何准确性分数 (权重: 0.3)  
            if 'geometric_accuracy' in results and isinstance(results['geometric_accuracy'], (int, float)):  
                scores.append(results['geometric_accuracy'])  
                weights.append(0.3)  
            
            if not scores:  
                return 0.0  
            
            # 归一化权重  
            weights = np.array(weights)  
            weights = weights / np.sum(weights)  
            
            # 计算加权平均  
            overall_score = np.sum(np.array(scores) * weights)  
            
            return float(overall_score)  
        except Exception as e:  
            print(f"Error computing overall score: {str(e)}")  
            return 0.0