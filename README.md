1. 初始化 `StockPrediction` 物件：

```python
aa = StockPrediction()
```

2. 輸入用於訓練的股票數據：

```python
aa.input_data('2330', '2012/01/01', '2023/01/01')
```

3. 選擇標準化方法：

```python
aa.choose_normalization('standard')
```

4. 選擇機器學習模型：

```python
aa.choose_model('SVR')
```

5. 訓練模型：

```python
model_path = aa.train_model(id)
```

6. 使用驗證方法 1 進行預測並生成圖形：

```python
aa.pre(id, '0050', '2023/05/29', '2023/06/02', 'standard')
```

7. 使用驗證方法 2 進行預測：

```python
predicted_price = aa.pre1(id, '0050', 'standard')
```

