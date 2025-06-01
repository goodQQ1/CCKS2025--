import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import nltk
import os
import ssl
import urllib.request
import zipfile
import shutil
import logging
from pathlib import Path


# ==================== 配置日志 ====================
def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('aigc_detector.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ==================== NLTK数据初始化 ====================
def initialize_nltk_resources():
    """初始化所有必需的NLTK资源"""
    # 设置自定义NLTK数据目录
    nltk_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.append(nltk_dir)

    # 需要下载的资源包
    required_resources = {
        'punkt': {
            'description': '基础分词器',
            'path': ['tokenizers', 'punkt']
        },
        'stopwords': {
            'description': '停用词表',
            'path': ['corpora', 'stopwords']
        },
        'punkt_tab': {
            'description': '分词器附加数据',
            'path': ['tokenizers', 'punkt_tab']
        },
        'averaged_perceptron_tagger': {
            'description': '词性标注器',
            'path': ['taggers', 'averaged_perceptron_tagger']
        }
    }

    # 禁用SSL验证（解决下载问题）
    ssl._create_default_https_context = ssl._create_unverified_context

    # 检查并下载资源
    for resource, info in required_resources.items():
        target_dir = os.path.join(nltk_dir, *info['path'])
        try:
            nltk.data.find(os.path.join(*info['path']))
            logger.info(f"[NLTK] 资源已存在: {resource} ({info['description']})")
        except LookupError:
            logger.info(f"[NLTK] 正在下载资源: {resource} ({info['description']})")
            try:
                nltk.download(resource, download_dir=nltk_dir)
                logger.info(f"[NLTK] 成功下载: {resource}")
            except Exception as e:
                logger.error(f"[NLTK] 下载失败 {resource}: {str(e)}")
                raise


# 初始化NLTK资源
initialize_nltk_resources()


# ==================== 主检测器类 ====================
class AIGCTextDetector:
    def __init__(self):
        self.tfidf = None
        self.model = None
        self.stop_words = self._load_stopwords()

    def _load_stopwords(self):
        """加载停用词表"""
        try:
            stop_words = set(stopwords.words('english'))
            logger.info("停用词表加载成功")
            return stop_words
        except Exception as e:
            logger.error(f"停用词表加载失败: {str(e)}")
            raise

    def load_data(self, file_path):
        """加载JSONL格式的数据"""
        texts = []
        labels = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        texts.append(data['text'])
                        if 'label' in data:
                            labels.append(data['label'])
                    except json.JSONDecodeError:
                        logger.warning(f"JSON解析错误 (行 {line_num}): 跳过该行")
                        continue
            return (texts, labels) if labels else (texts, None)
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return None, None

    def preprocess_text(self, text):
        """文本预处理管道"""
        if not isinstance(text, str):
            return ""

        try:
            # 转换为小写
            text = text.lower()
            # 移除URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # 移除HTML标签
            text = re.sub(r'<.*?>', '', text)
            # 移除标点符号
            text = text.translate(str.maketrans('', '', string.punctuation))
            # 移除数字
            text = re.sub(r'\d+', '', text)
            # 分词
            tokens = word_tokenize(text)
            # 移除停用词和短词
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"文本预处理失败: {str(e)}")
            return ""

    def extract_features(self, texts, fit=False):
        """提取TF-IDF特征"""
        try:
            if fit:
                self.tfidf = TfidfVectorizer(
                    max_features=15000,
                    ngram_range=(1, 3),
                    stop_words='english',
                    sublinear_tf=True,
                    min_df=5,
                    max_df=0.7
                )
                return self.tfidf.fit_transform(texts)
            return self.tfidf.transform(texts)
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}")
            raise

    def train_model(self, X_train, y_train):
        """训练并选择最佳模型"""
        models = {
            'LogisticRegression': LogisticRegression(
                C=0.5, penalty='l2', solver='liblinear',
                max_iter=1000, random_state=42, class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_split=5, random_state=42,
                n_jobs=-1, class_weight='balanced'
            ),
            'SVM': SVC(
                C=1.0, kernel='linear',
                probability=True, random_state=42,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1,
                max_depth=5, random_state=42
            )
        }

        best_score = 0
        best_model = None

        for name, model in models.items():
            try:
                logger.info(f"训练模型: {name}")
                model.fit(X_train, y_train)
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5, scoring='f1', n_jobs=-1
                )
                mean_score = np.mean(scores)
                logger.info(f"{name} 交叉验证F1分数: {mean_score:.4f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    logger.info(f"当前最佳模型: {name} (F1: {best_score:.4f})")
            except Exception as e:
                logger.error(f"模型 {name} 训练失败: {str(e)}")
                continue

        if best_model is None:
            raise ValueError("所有模型训练失败")

        self.model = best_model
        logger.info(f"\n最终选择模型: {type(best_model).__name__}, F1分数: {best_score:.4f}")
        return best_model

    def evaluate(self, X, y_true):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("模型未训练")

        y_pred = self.model.predict(X)
        report = classification_report(y_true, y_pred)
        logger.info("\n分类报告:\n" + report)
        return f1_score(y_true, y_pred)

    def predict(self, texts):
        """预测新文本"""
        preprocessed = [self.preprocess_text(text) for text in texts]
        features = self.extract_features(preprocessed)
        return self.model.predict(features)

    def save_predictions(self, predictions, file_path):
        """保存预测结果"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(map(str, predictions)))
            logger.info(f"预测结果已保存到: {file_path}")
        except Exception as e:
            logger.error(f"结果保存失败: {str(e)}")
            raise


# ==================== 主函数 ====================
def main():
    try:
        logger.info("=" * 60)
        logger.info("AI生成文本检测系统启动".center(50))
        logger.info("=" * 60)

        detector = AIGCTextDetector()

        # 配置路径（可根据需要修改）
        base_dir = os.path.join(os.path.expanduser('~'), 'Desktop')
        train_path = os.path.join(base_dir, 'train', 'train.jsonl')
        test_path = os.path.join(base_dir, 'test_521', 'test.jsonl')
        output_path = os.path.join(base_dir, 'submit.txt')

        # 1. 加载训练数据
        logger.info(f"\n加载训练数据: {train_path}")
        train_texts, train_labels = detector.load_data(train_path)
        if not train_texts:
            raise ValueError("训练数据加载失败")

        # 2. 预处理和特征工程
        logger.info("\n文本预处理中...")
        preprocessed_train = [detector.preprocess_text(text) for text in train_texts]
        logger.info(f"预处理完成，有效样本数: {len(preprocessed_train)}")

        logger.info("特征提取中...")
        X_train = detector.extract_features(preprocessed_train, fit=True)
        y_train = np.array(train_labels)
        logger.info(f"特征矩阵形状: {X_train.shape}")

        # 3. 训练模型
        logger.info("\n开始模型训练...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        detector.train_model(X_train, y_train)

        # 4. 评估模型
        logger.info("\n模型评估中...")
        val_f1 = detector.evaluate(X_val, y_val)
        logger.info(f"验证集F1分数: {val_f1:.4f}")

        # 5. 预测测试集
        logger.info(f"\n加载测试数据: {test_path}")
        test_texts, _ = detector.load_data(test_path)
        if not test_texts:
            raise ValueError("测试数据加载失败")

        logger.info("生成预测结果...")
        test_pred = detector.predict(test_texts)
        logger.info(f"预测完成，样本数: {len(test_pred)}")

        # 6. 保存结果
        detector.save_predictions(test_pred, output_path)

        logger.info("\n" + "=" * 60)
        logger.info("处理完成！".center(50))
        logger.info(f"结果文件: {output_path}".center(50))
        logger.info("=" * 60)

    except Exception as e:
        logger.error("程序运行出错:", exc_info=True)
        input("按Enter键退出...")
        return 1

    return 0


if __name__ == "__main__":
    # 检查NLTK资源是否完整
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError as e:
        logger.warning(f"缺少NLTK资源: {str(e)}")
        logger.info("正在尝试自动下载...")
        initialize_nltk_resources()

    exit_code = main()
    exit(exit_code)