# Rubik-s-Cube
University Graduation Project

# 待辦事項

- [ ] 補上 requirement.txt
- [ ] 完成 config.py

#  報告

## Title

基於機器學習的狀態空間搜尋演算法

## Abstract

​	本報告為一份課餘自主研究成果。報告主要聚焦於「基於機器學習的狀態空間搜尋演算法」這一主題，並展示了學生在此領域的研究能力和熱情。報告不僅包括對問題背景的深入理解和分析，也詳述了實驗設計和過程。

​	報告的架構上，我首先介紹了我所遇到的問題與其相關的背景知識，爾後說明了我對於問題的理解與分析，雖然這前面的篇章並沒有著墨在關於實際開發或是程式碼方向的討論，但我認為對於問題的認識和分析是做研究的基石，亦是實作經驗的一環。報告後段則詳細描述了實驗設計和過程，相關的程式碼和資源已上傳至 GitHub 儲存庫以供參考[^1]。

​	主題則是探討了基於機器學習的狀態空間搜尋演算法，特別是在解決 Rubik's Cube 這類問題上。報告首先回顧了現有的狀態空間搜尋演算法，如 BFS 和 DFS，以及它們的局限性。接著，介紹了由學生設想的一種新的 Encoder-Decoder 架構，該架構旨在將離散的狀態空間映射到一個連續的空間中，以便進行更有效的搜尋。研究中遇到的主要挑戰是如何設計一個有效的 Discriminator，以確保在連續空間中的映射能夠反映出狀態間的實際距離。學生也嘗試了多種方法，包括使用高維空間和球面空間，但目前尚未找到一個完全有效的解決方案。

​	儘管如此，這項研究仍提供了一個新的角度來看待狀態空間搜尋問題，相信為類似的研究提供了有價值的洞見。ㄐㄐ

## Introduction

​	在認識探索狀態空間搜尋演算法時，對於演算法設計感受到困惑。當下常見的演算法多基於 BFS 或 DFS 的無信息搜尋演算法。已知對排序數列而言，二分搜尋法能優化搜尋演算法，而狀態空間多為固定且明確，是否也有相似的優化途徑？

​	狀態空間搜尋演算法優化多依賴剪枝或先驗知識如啟發式設計等。而其中一種優化方法，雙向搜尋透過增加起始搜尋狀態的基數以減少搜尋節點，能將時間複雜度從 $O(b^d)$ 降至 $O(b^{d/2})$。

![](https://hackmd.io/_uploads/BJ77JjsV2.png)

​	若能擴展起始節點至 n 個狀態，時間複雜度可降至 $O(b^{d/n})$。主要挑戰在於對問題的狀態空間結構理解不足，無法設置有效的額外起始節點。故提出一機器學習方法，嘗試通過訓練神經網路及解碼器映射狀態空間，探索狀態空間結構的目的是為了將狀態映射到連續空間中，並利用插值法增加有效的起始狀態的節點，減少總搜尋節點以達到優化的目的。

<img src="https://hackmd.io/_uploads/HJ_3mjjEn.png" />

## Related Works

### DeepCubeA[^2]

​	一種深度強化學習演算法，其中以 Rubik's Cube 為例子。該演算法使用 Approximate Value Iteration 的方法來訓練一個 Deep Neural Network ，該神經網路可以估計達到目標的成本，也被稱之為 cost-to-go 函數。這個神經網路將作為 A* 搜尋演算法的啟發式，指導搜尋過程從而找出問題的解。

**DeepCubeA 的問題點**

- **可解釋性**
	由於 DeepCubeA 的 Value Function 訓練過程缺乏透明度，它在選擇下一步的動作時缺乏直觀的解釋性。因此，它不能直接作為決策標準，而是通過與A*搜索算法結合，將神經網路作為啟發式的估計值來使用。 

- **過多的運算資源消耗**
	作為A*搜索算法的啟發式，可以推論出神經網路的推論次數將會隨著狀態空間的擴大而增加，即便在計算時間複雜度上將神經網路視為常數，它本質上仍是一個龐大的深度網路，對於時間複雜度和計算資源的需求不容忽視。

### Kociemba's God's Algorithm[^3]

​	既然以  Rubik's Cube 為例子，那就不得不提及 Herbert Kociemba 的 God's Algorithm 。這是一個著名的演算法，旨在以最少的旋轉次數解決 Rubik's Cube。其算法的核心思想是將解決 Rubik's Cube 的過程分為兩個主要階段，以減少所需的搜尋和計算量。 

- **階段1**：在第一階段，算法將立方體簡化到一個較小的子群，通常通過限制允許的移動來實現。這可以看作是一個粗糙的搜索，旨在將立方體移到一個接近解決方案的狀態。  
- **階段2**：在第二階段，算法使用完整的移動集來解決從階段1結果到完全解決的立方體。這個階段涉及更精細的搜索，以最小化所需的旋轉次數。 

**Kociemba's God's Algorithm 的缺點問題點**

- **計算複雜度**：雖然兩階段算法有效地減少了計算量，但由於Rubik's Cube的狀態空間非常大，算法仍然需要相當大的計算資源。特別是在階段2中，搜索空間可能仍然非常大，導致計算時間長和記憶體使用量高。
- **可擴展性**：Kociemba的算法主要針對Rubik's Cube設計。將它擴展到其他類型的拼圖或具有不同狀態空間結構的問題可能會很困難，需要大量的自定義和調整。

## Problem Definition and Objectives

### Problem to be Solved

​	本研究旨在設計一套基於機器學習的演算法，以優化狀態空間搜尋演算法。傳統的狀態空間搜尋演算法，如廣度優先搜索（BFS）或深度優先搜索（DFS），在面對大規模或高維度的狀態空間時，往往會遭遇計算資源的限制。儘管有一些優化技術，如剪枝和啟發式搜索，但它們通常依賴於問題的先驗知識或特定結構。因此，本研究旨在探索如何利用機器學習方法來自動化地學習和理解狀態空間的結構，從而優化搜尋演算法的效率和效果，而在實驗上我會以 Rubik's Cube 的問題為主。

### Specific Objectives

- **Encoder-Decoder 架構設計**：
	- 本研究計劃設計一個 Encoder-Decoder 的架構。Encoder 的目的是將離散的狀態空間映射到一個連續的空間中，而 Decoder 則負責將連續空間中的點映射回原始的狀態空間。這種映射機制旨在提供一種方式來計算和理解不同狀態之間的關係，並探索狀態空間的潛在結構。
- **演算法的設計**：
	- 利用上述的 Encoder-Decoder 架構和狀態空間結構的理解，本研究將採用插值法來計算狀態間的中間點。通過找到連續空間中的中間點，並使用 Decoder 將這些點映射回原始的狀態空間，本研究旨在增加有效的起始節點，從而減少搜尋過程中需要探索的節點總數。這項技術有望提高搜尋演算法的效率，並降低計算資源的需求。

​	通過這兩個主要目標，本研究旨在提供一種新的、基於機器學習的方法來優化狀態空間搜尋演算法，並為解決大規模或高維度狀態空間中的問題提供一種有效的解決方案或是額外的思路。

## Methodology

### 模型架構

​	為了達成將狀態映射至連續空間中並重建回狀態的作業可以透過 AutoEncoder 架構完成，但同時我也希望 latent space 具備能夠將相近的狀態的距離表現出來的一個空間中。因此借鑒 Adversarial AutoEncoder的架構[^4]，其引入 Discriminator 來試圖約束 Encoder 的輸出的分佈趨近某種目標分佈，我也可以透過訓練一個神經網路輸入生成的 latent space 並以狀態的實際距離作為 label 進行訓練並同時約束 Encoder 的生成，以此達成約束 Encoder 生成符合能透過 latent space 計算距離的結果。更進一步說，若是以歐式距離公式作為 Discriminator ，則可以達成透過 latent space 逕直計算距離的目的。架構如下圖。

<img src="https://hackmd.io/_uploads/rysBEisEn.png" style="zoom: 67%;" />

而這是模型訓練過程的虛擬碼，實際上我以 Pytorch 撰寫之。

```python
Initialize Hyperparameter

Initialize Encoder, Decoder, Discriminator models
Initialize Adam optimizers for Encoder, Decoder, Discriminator
Initialize learning rate schedulers for Encoder, Decoder, Discriminator

Initialize MSE Loss function

Load training data from "./dataset" into DataLoader

FOR each epoch in number of epochs:
    FOR each batch in DataLoader:
        Clear gradients for Encoder and Decoder optimizers

        /* Encoding Phase */
        Compute initial_state_bottleneck using Encoder
        Compute destination_state_bottleneck using Encoder

        /* Decoding Phase */
        Compute initial_state_reconstructed using Decoder
        Compute destination_state_reconstructed using Decoder

        Compute reconstruction_loss using MSE Loss

        /* Distance Calculation */
        Compute distance_fake using initial_state_bottleneck and destination_state_bottleneck
        Compute distance_loss using MSE Loss between distance_fake and actual distance

        /* Combined Loss */
        Compute combined_loss = reconstruction_loss + lambda * distance_loss

        Perform backward pass on combined_loss
        Update Encoder and Decoder parameters

    Update learning rate using scheduler for Encoder, Decoder, and Discriminator

Save model weights for Encoder, Decoder, Discriminator
```

### 資料的分析與特性

​	Rubik's Cube 的狀態空間十分龐大，大約有 $4 \times 10^{19}$ 個狀態。其中，根據 Richard E. Korf 的研究顯示，在所有狀態中與自己相差 18 個單位距離的狀態是最多的，如下表所述。補充說明的是，3 * 3 Rubik's Cube 被驗證出來的 God's number 為 20[^5]，抑是說在所有狀態中相距最大為 20 個單位。 

​	資料集的格式將為目標狀態、起點狀態與距離的 tuple。為了生成準確的距離值而非近似解，我使用了 Herbert Kociemba 開發的 RubiksCube-OptimalSolver [^6]。對方指出，在配備 AMD Ryzen 7 3700X 3.59 GHz 的機器上實驗，The optimal solving time was in a range between 37 s and 1167 s, the total time for the 10 cubes was 3841 s. The average optimal solving length was 17.80。這也意味著光是生成一筆測資就需要將近 6 分鐘，而若是生成到一筆距離為 19 或 20 的資料，則需要花上 3 到 7 個小時，這對於機器學習訓練來說是非常致命的問題。而我的解決方法則是讓 Solver 在超過我機器計算 17 個距離單位的平均時間時自動停止，並皆假設為 18 個距離單位，由於 19 與 18 距離單位誤差並不會影響太多且 20 距離單位在機率上近乎不常出現，這方法生成的資料集品質並不會差到哪裡。同時這為解決資料集生成的速度提高了 6 倍之多。我共生成了20萬筆資料集，下圖是生成的資料集範例。

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20231016221641741.png" alt="image-20231016221641741" style="zoom:67%;" />

​	補充說明，一筆測資裡面是有兩個目標狀態的，其中一個是距離較遠的狀態，另一個是距離較近的狀態。這是實驗初期訓練 Distance Loss 的效果一直都不好，嘗試使用 Contrastive Learning 的方法，將相近的狀態拉近、將較遠的距離拉遠，是否能更快的收斂並取得較好的效果時做的修正。在實驗後期，我將實驗問題改為 2 * 2 的 Rubik's Cube，其狀態空間總數僅有 3,674,160 個，且 God's number 為 11。由於原先問題中的資料集生成速度慢，因此為了重複實驗才儲存起來，而 2 * 2 的 Rubik's Cube 並不需要這麼做，我選擇以強化學習相似的作法，在訓練過程中一併生成，解決了資料量不足的問題。方法透過將資料集生成包裝在 Pytorch 的 Dataset Class 裡面來實現。

## **Experimental Design**

​	實驗上，對於狀態重建的 Encoder-Decoder 架構的訓練上在經過對狀態使用 One hot encoding 的 preprocess 後可以很容易的完成任務，因此我以如何設計 Discriminator 為主要的實驗方向。

### Discriminator 的設計

​	狀態空間是離散的，若要將其映射到連續空間中無疑是困難的。該研究方向主要是研究如何設計 Discriminator，在研究初期我以神經網路作為 Discriminator。然而該設計會導致整個模型遲遲無法收斂，並且使用神經網路計算映射到 latent space 後的距離，與原先希望映射到連續空間的想法相悖，因此在這之後我皆以空間的距離公式作為 Discriminator。一開始，我選擇使用歐氏距離作為 Discriminator。然而，這種方法在實驗中表現不佳：Distance Loss 最初下降得非常迅速，但很快就達到一個瓶頸，並在這個值附近波動。

​	 為了解決這個問題，我也尋求了許多人的意見。得出以下結論：大多數的魔術方塊需要 18 步才能解開，因此大部分的狀態應該大致位於一個半徑為 18 的球體“表面”上（如果已解開的狀態位於座標空間的中心）。但是，如果你選擇這個球體上的一個點，那麼到這個球體上大多數其他方塊的距離對於大多數方塊來說，必須再次是 18。這結論也啟發了我，這特性導致魔術方塊的狀態是無法在歐式空間中表示出來的。也因此在那之後我嘗試使用了其他方式來訓練，例如球面空間等等，因為球面空間可以具備上述的特性，實驗方法則是將 Latent space 假定為球座標系的表示，並求測地線距離等等。其中比較有趣的是我以拉高 Latent space 的維度的實驗。

#### 高維空間

透過觀察發現，高維空間有著一些有趣的特性。例如越高維度，座標點越分佈於邊界體積、對於隨機點對而言，其會越接近某個特定值。如下圖所示。這與上述分析出來的魔術方塊特性一致，雖然這是由於維度詛咒的問題，卻也是一個可以嘗試的做法。

![image-20231016221456586](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20231016221456586.png)

## **Results and Discussion**

### 結果

​	測試方法如下，我們能夠透過將兩個 Rubik's Cube 輸入 Encoder 並得到兩個 Latent space 的表示。我們可以透過公式計算中點座標，並由 Decoder 將其映射回原始狀態。

<img src="https://hackmd.io/_uploads/S1ma4siVn.png" style="zoom: 33%;" />

<img src="https://hackmd.io/_uploads/S1OK4iiNh.png" style="zoom:65%;" />

​	然而，在目前階段還無法有效的訓練出映射，這取決於 Discriminator 能否有效的約束 Encoder。現階段我以 Encoder 的映射程度作為檢測方法。測試方法如下，我分別輸入以解決的魔術方塊序列將其 Encode，並輸入 Discriminator 來計算與目標狀態的距離，由下圖可以看到，雖然並沒有呈現精確的距離的數值，但所計算出來的數值仍然是有序的。緊接著，我取得了起始狀態和目標狀態的其他狀態，並求他們與起始狀態和目標狀態的那條直線的距離，由下圖可見，大部分的狀態所映射到的位置仍然距離路徑的值線有些許的誤差，並以越距離遠的狀態距離的越多。這也是目前即使求出兩點映射卻無法從中間節點 Decode 回原始狀態的最大問題。

![image-20231016221531930](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20231016221531930.png)

​	然而縱使神經網路並沒辦法很好的將狀態映射到某個維度的歐幾里得空間中，我們或許也可以證明出當 distance(AC) + distance(CB) - distance(AB) 小於某個值的時候，在做多點搜尋時的路徑也會包含最優路徑。如此一來，我們只要將搜尋的節點再遍歷一次就能得到最優路徑，時間複雜度並不變，同時也能達到有效減少搜尋節點的目的。

![image-20231016221352480](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20231016221352480.png)

## **Conclusion**

本研究報告主要探討了基於機器學習的狀態空間搜尋演算法，特別是在 Rubik's Cube 的問題領域。透過 Encoder-Decoder 架構和 Discriminator 的設計，我嘗試了多種方法來映射狀態空間到一個連續的空間中，並進一步優化狀態空間搜尋演算法。

在實驗設計階段，採用了多種不同的 Discriminator 設計，包括使用歐氏距離和高維空間的特性。然而，目前的結果顯示，儘管這些方法在理論上具有潛力，但在應用中仍面臨一些挑戰，尤其是在 Discriminator 的設計和訓練過程中。這些挑戰主要源於 Rubik's Cube 狀態空間的特殊性質，這使得將其映射到一個連續的空間中變得相對困難。儘管如此，這項研究仍然提供了一個新的視角和方法來探索狀態空間搜尋演算法的優化，大概吧。

## **References**

[^1]: https://github.com/eason280711/Rubik-s-Cube
[^2]: https://deepcube.igb.uci.edu/
[^3]: https://github.com/hkociemba
[^4]: https://arxiv.org/abs/1511.05644
[^5]: https://www.cube20.org/
[^6]: https://github.com/hkociemba/RubiksCube-OptimalSolver

