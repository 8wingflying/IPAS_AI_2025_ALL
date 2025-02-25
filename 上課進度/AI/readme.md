# AI
- 主題
  - 電腦視覺(Computer vision)
  - 自然語言處理(NLP|Natural Language Processing)
  - 生成式 AI
  - 多模態AI
- 模型與技術
  - CNN
  - RNN
  - GAN  
  - RL(強化學習) 
# CNN
- 1.CNN範例示範
  - [Convolutional Neural Network (CNN)| Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)
  - 【TensorFlow】[官方範例: Convolutional Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn)
  - 【TensorFlow】[官方範例: Image classification](https://www.tensorflow.org/tutorials/images/classification)
    - [tf.keras.utils.image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)
- 2.CNN核心元件
- 3.[CNN model經典模型](./CNNModels.md) 

- 4.Transfer Learning遷移學習
  - 圖像分類模型有數百萬個參數。從頭訓練需要大量帶標籤的訓練資料和強大的算力。
  - 遷移學習會在新模型中重複使用已在相關任務上訓練過的模型的一部分，可以顯著降低這些需求。
  - 兩大類方式:(1)特徵萃取 (2)微調法(Fine-tuning)
  - 【TensorFlow】[官方範例: Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
  - 【TensorFlow】[官方範例: Transfer learning with TensorFlow Hub](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)
    - 使用TensorFlow Hub的pre-trained model:
