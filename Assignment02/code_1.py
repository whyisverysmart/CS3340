import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess_single_image(img_path, target_size=(40, 20)):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, (target_size[1], target_size[0]))
        # 二值化
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

def preprocess_and_save_images(input_folder='./Task1', output_folder='./ProcessedTask1', target_size=(40, 20)):
    for i in range(1, 11):
        filename = f"{i:02d}.png"
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        binary = preprocess_single_image(input_path, target_size)
        cv2.imwrite(output_path, binary)

def extract_edge_orientation_histogram(img, bins=8):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    # 只统计边缘位置（通过Canny筛选边缘点）
    edge_mask = cv2.Canny(img, 100, 200) > 0
    angle = angle[edge_mask]
    # 构建方向直方图
    hist, _ = np.histogram(angle, bins=bins, range=(0, 360), density=True)
    return hist

def extract_sift_descriptors(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors  # descriptors 是 N×128 的矩阵，N 是关键点个数

def extract_projection_features(img):
    vertical = np.sum(img == 255, axis=0)
    horizontal = np.sum(img == 255, axis=1)
    return np.concatenate([horizontal, vertical])  # 返回长度60的向量


def compare_vectors(f1, f2, method='cosine'):
    if method == 'cosine':
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1 - np.dot(f1, f2) / (norm1 * norm2)
    elif method == 'euclidean':
        return np.linalg.norm(f1 - f2)


def build_features_general(feature_type):
    ###
    # Only for numbers and letters
    ###
    gallery_features = {}  # key: label, value: mean feature vector
    gallery_path='./chars'

    for label in os.listdir(gallery_path):
        if label == '__Hanzi':
            continue
        label_path = os.path.join(gallery_path, label)
        features = []

        for fname in os.listdir(label_path):
            img_path = os.path.join(label_path, fname)
            binary = preprocess_single_image(img_path)

            if feature_type == 'edge_hist':
                # 特征维度为8
                feat = extract_edge_orientation_histogram(binary)
            elif feature_type == 'projection':
                # 特征维度为60
                feat = extract_projection_features(binary)
            elif feature_type == 'sift':
                # 特征维度为128
                descriptors = extract_sift_descriptors(binary)
                if descriptors is not None:
                    feat = np.mean(descriptors, axis=0)
                else:
                    feat = np.zeros(128)

            features.append(feat)
        if features:
            mean_feature = np.mean(features, axis=0)
            gallery_features[label] = mean_feature

    return gallery_features

def build_features_general_2(feature_type):
    ###
    # Only for numbers and letters
    ###
    gallery_features = {}  # key: label, value: list of feature vectors
    gallery_path = './chars'

    for label in os.listdir(gallery_path):
        if label == '__Hanzi':
            continue
        label_path = os.path.join(gallery_path, label)
        features = []

        for fname in os.listdir(label_path):
            img_path = os.path.join(label_path, fname)
            binary = preprocess_single_image(img_path)

            if feature_type == 'edge_hist':
                feat = extract_edge_orientation_histogram(binary)
            elif feature_type == 'projection':
                feat = extract_projection_features(binary)
            elif feature_type == 'sift':
                descriptors = extract_sift_descriptors(binary)
                if descriptors is not None:
                    feat = np.mean(descriptors, axis=0)
                else:
                    continue
            features.append(feat)

        if features:
            gallery_features[label] = features

    return gallery_features

def build_features_hanzi_2(feature_type):
    ###
    # Only for Hanzi
    ###
    gallery_features = {}  # key: label, value: list of feature vectors
    gallery_path = './chars/__Hanzi'

    for label in os.listdir(gallery_path):
        label_path = os.path.join(gallery_path, label)
        features = []

        for fname in os.listdir(label_path):
            img_path = os.path.join(label_path, fname)
            binary = preprocess_single_image(img_path)

            if feature_type == 'edge_hist':
                feat = extract_edge_orientation_histogram(binary)
            elif feature_type == 'projection':
                feat = extract_projection_features(binary)
            elif feature_type == 'sift':
                descriptors = extract_sift_descriptors(binary)
                if descriptors is not None:
                    feat = np.mean(descriptors, axis=0)
                else:
                    continue
            features.append(feat)

        if features:
            gallery_features[label] = features

    return gallery_features

def recognize_general_single(img_path, gallery_features, feature_type='projection', sim_method='cosine'):
    binary = preprocess_single_image(img_path)

    if feature_type == 'edge_hist':
        test_feat = extract_edge_orientation_histogram(binary)
    elif feature_type == 'projection':
        test_feat = extract_projection_features(binary)
    elif feature_type == 'sift':
        descriptors = extract_sift_descriptors(binary)
        if descriptors is not None:
            test_feat = np.mean(descriptors, axis=0)
        else:
            raise ValueError(f"No SIFT descriptors found in {img_path}")

    best_label = None
    best_score = float('inf')
    for label, mean_feat in gallery_features.items():
        score = compare_vectors(test_feat, mean_feat, method=sim_method)
        if score < best_score:
            best_score = score
            best_label = label
    return best_label

def recognize_general_single_2(img_path, gallery_features, feature_type='projection', sim_method='cosine', agg_method='min'):
    binary = preprocess_single_image(img_path)

    if feature_type == 'edge_hist':
        test_feat = extract_edge_orientation_histogram(binary)
    elif feature_type == 'projection':
        test_feat = extract_projection_features(binary)
    elif feature_type == 'sift':
        descriptors = extract_sift_descriptors(binary)
        if descriptors is not None:
            test_feat = np.mean(descriptors, axis=0)
        else:
            raise ValueError(f"No SIFT descriptors found in {img_path}")

    best_label = None
    best_score = float('inf')
    # score = {}
    for label, feature_list in gallery_features.items():
        scores = [compare_vectors(test_feat, feat, method=sim_method) for feat in feature_list]
        
        if agg_method == 'mean':
            final_score = np.mean(scores)
        elif agg_method == 'min':
            final_score = np.min(scores)
        else:
            raise ValueError(f"Unsupported agg_method: {agg_method}")
        # score[label] = final_score

        if final_score < best_score:
            best_score = final_score
            best_label = label
    
    # plot_similarity_histogram(score)

    return best_label

def plot_similarity_histogram(score_dict):
    labels = list(score_dict.keys())
    scores = list(score_dict.values())

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(sorted_labels, sorted_scores, color='skyblue', edgecolor='black')
    plt.xlabel("Chars")
    plt.ylabel("Similarity")

    for bar, score in zip(bars, sorted_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", 
                 ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig("similarity_hist.png")
    plt.close()

def recognize_general(gallery_features=None, feature_type='projection', sim_method='cosine'):
    results = {} # key: filename, value: predicted label
    processed_folder = './Task1'
    for i in range(2, 10): # 01 and 09 are Hanzi
        filename = f"{i:02d}.png"
        input_path = os.path.join(processed_folder, filename)
        pred = recognize_general_single(input_path, gallery_features, feature_type, sim_method)
        results[filename] = pred
    return results

def recognize_general_2(gallery_features=None, feature_type='projection', sim_method='cosine'):
    ###
    # Only for numbers and letters
    ###
    results = {} # key: filename, value: predicted label
    processed_folder = './Task1'
    # for i in range(2, 10): # 01 and 09 are Hanzi
    for i in [2]:
        filename = f"{i:02d}.png"
        input_path = os.path.join(processed_folder, filename)
        pred = recognize_general_single_2(input_path, gallery_features, feature_type, sim_method)
        results[filename] = pred
    return results

def recognize_hanzi_2(gallery_features=None, feature_type='projection', sim_method='cosine'):
    ###
    # Only for Hanzi
    ###
    results = {} # key: filename, value: predicted label
    processed_folder = './Task1'
    for i in [1, 9]: # 01 and 09 are Hanzi
        filename = f"{i:02d}.png"
        input_path = os.path.join(processed_folder, filename)
        pred = recognize_general_single_2(input_path, gallery_features, feature_type, sim_method)
        results[filename] = pred
    return results

if __name__ == "__main__":
    # preprocess_and_save_images()

    # 维数为60
    general_projection_features = build_features_general_2('projection')
    # general_projection_features = build_features_hanzi_2('projection')
    # 维数为8
    # general_edge_hist_features = build_features_general('edge_hist')
    # general_edge_hist_features = build_features_general_2('edge_hist')
    # 维数为128
    # general_sift_features = build_features_general('sift')
    # general_sift_features = build_features_general_2('sift')

    general_projection_results = recognize_general_2(general_projection_features, feature_type='projection', sim_method='cosine')
    # general_projection_results = recognize_hanzi_2(general_projection_features, feature_type='projection', sim_method='cosine')
    # general_edge_hist_results = recognize_general(general_edge_hist_features, feature_type='edge_hist', sim_method='euclidean')
    # general_edge_hist_results = recognize_general_2(general_edge_hist_features, feature_type='edge_hist', sim_method='euclidean')
    # general_sift_results = recognize_general(general_sift_features, feature_type='sift', sim_method='euclidean')
    # general_sift_results = recognize_general_2(general_sift_features, feature_type='sift', sim_method='euclidean')
    
    print("Projection Results:")
    for img_name, pred in general_projection_results.items():
        print(f"{img_name} -> {pred}")
    # print("Edge Histogram Results:")
    # for img_name, pred in general_edge_hist_results.items():
    #     print(f"{img_name} -> {pred}")
    # print("SIFT Results:")
    # for img_name, pred in general_sift_results.items():
    #     print(f"{img_name} -> {pred}")