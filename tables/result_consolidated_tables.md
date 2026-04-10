## Table 1. NER results (F1 Macro)

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Language</th>
      <th rowspan="2">Baseline</th>
      <th colspan="10">NeFT</th>
      <th colspan="10">Probeless</th>
    </tr>
    <tr>
      <th>5</th><th>10</th><th>15</th><th>20</th><th>25</th><th>30</th><th>35</th><th>40</th><th>45</th><th>50</th>
      <th>5</th><th>10</th><th>15</th><th>20</th><th>25</th><th>30</th><th>35</th><th>40</th><th>45</th><th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mBERT</td><td>Arabic</td><td>0.4985</td>
      <td>0.5787</td><td>0.5622</td><td>0.5918</td><td>0.5575</td><td>0.5870</td><td>0.5849</td><td>0.5840</td><td>0.5410</td><td>0.5236</td><td>0.5584</td>
      <td>0.5473</td><td>0.5374</td><td>0.5613</td><td>0.5592</td><td>0.5676</td><td>0.5378</td><td>0.6340</td><td>0.5306</td><td>0.5705</td><td>0.5877</td>
    </tr>
    <tr>
      <td>mBERT</td><td>Chinese</td><td>0.8651</td>
      <td>0.8672</td><td>0.8524</td><td>0.8711</td><td>0.8644</td><td>0.8624</td><td>0.8634</td><td>0.8726</td><td>0.8699</td><td>0.8709</td><td>0.8582</td>
      <td>0.8674</td><td>0.8634</td><td>0.8607</td><td>0.8559</td><td>0.8611</td><td>0.8659</td><td>0.8542</td><td>0.8562</td><td>0.8624</td><td>0.8636</td>
    </tr>
    <tr>
      <td>mBERT</td><td>French</td><td>0.9295</td>
      <td>0.9277</td><td>0.9257</td><td>0.9262</td><td>0.9283</td><td>0.9248</td><td>0.9204</td><td>0.9262</td><td>0.9294</td><td>0.9234</td><td>0.9230</td>
      <td>0.9227</td><td>0.9135</td><td>0.9340</td><td>0.9221</td><td>0.9244</td><td>0.9275</td><td>0.9238</td><td>0.9333</td><td>0.9285</td><td>0.9238</td>
    </tr>
    <tr>
      <td>mBERT</td><td>Hindi</td><td>0.9327</td>
      <td>0.9299</td><td>0.9407</td><td>0.9332</td><td>0.9317</td><td>0.9326</td><td>0.9344</td><td>0.9399</td><td>0.9335</td><td>0.9281</td><td>0.9356</td>
      <td>0.9336</td><td>0.9341</td><td>0.9397</td><td>0.9316</td><td>0.9351</td><td>0.9320</td><td>0.9205</td><td>0.9371</td><td>0.9346</td><td>0.9272</td>
    </tr>
    <tr>
      <td>XLM</td><td>Arabic</td><td>0.6184</td>
      <td>0.6172</td><td>0.5931</td><td>0.5779</td><td>0.6242</td><td>0.5756</td><td>0.5998</td><td>0.5944</td><td>0.5650</td><td>0.5799</td><td>0.5236</td>
      <td>0.5467</td><td>0.5297</td><td>0.5930</td><td>0.5782</td><td>0.6081</td><td>0.6458</td><td>0.6434</td><td>0.5844</td><td>0.6012</td><td>0.5167</td>
    </tr>
    <tr>
      <td>XLM</td><td>Chinese</td><td>0.8479</td>
      <td>0.8482</td><td>0.8566</td><td>0.8560</td><td>0.8484</td><td>0.8463</td><td>0.8575</td><td>0.8513</td><td>0.8520</td><td>0.8403</td><td>0.8481</td>
      <td>0.8429</td><td>0.8526</td><td>0.8445</td><td>0.8523</td><td>0.8549</td><td>0.8580</td><td>0.8544</td><td>0.8594</td><td>0.8528</td><td>0.8459</td>
    </tr>
    <tr>
      <td>XLM</td><td>French</td><td>0.9221</td>
      <td>0.9269</td><td>0.9179</td><td>0.9282</td><td>0.9161</td><td>0.9284</td><td>0.9264</td><td>0.9208</td><td>0.9197</td><td>0.9187</td><td>0.9218</td>
      <td>0.9250</td><td>0.9264</td><td>0.9206</td><td>0.9275</td><td>0.9175</td><td>0.9174</td><td>0.9277</td><td>0.9182</td><td>0.9280</td><td>0.9217</td>
    </tr>
    <tr>
      <td>XLM</td><td>Hindi</td><td>0.9401</td>
      <td>0.9333</td><td>0.9377</td><td>0.9347</td><td>0.9325</td><td>0.9387</td><td>0.9271</td><td>0.9341</td><td>0.9280</td><td>0.9438</td><td>0.9413</td>
      <td>0.9419</td><td>0.9399</td><td>0.9352</td><td>0.9424</td><td>0.9401</td><td>0.9368</td><td>0.9390</td><td>0.9385</td><td>0.9389</td><td>0.9307</td>
    </tr>
  </tbody>
</table>

## Table 2. POS results (F1 Macro)

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Language</th>
      <th rowspan="2">Baseline</th>
      <th colspan="10">NeFT</th>
      <th colspan="10">Probeless</th>
    </tr>
    <tr>
      <th>5</th><th>10</th><th>15</th><th>20</th><th>25</th><th>30</th><th>35</th><th>40</th><th>45</th><th>50</th>
      <th>5</th><th>10</th><th>15</th><th>20</th><th>25</th><th>30</th><th>35</th><th>40</th><th>45</th><th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mBERT</td><td>Arabic</td><td>0.9533</td>
      <td>0.9566</td><td>0.9522</td><td>0.9533</td><td>0.9497</td><td>0.9552</td><td>0.9494</td><td>0.9491</td><td>0.9564</td><td>0.9521</td><td>0.9564</td>
      <td>0.9526</td><td>0.9500</td><td>0.9554</td><td>0.9544</td><td>0.9583</td><td>0.9561</td><td>0.9553</td><td>0.9576</td><td>0.9552</td><td>0.9573</td>
    </tr>
    <tr>
      <td>mBERT</td><td>Chinese</td><td>0.9572</td>
      <td>0.9573</td><td>0.9584</td><td>0.9568</td><td>0.9568</td><td>0.9610</td><td>0.9579</td><td>0.9597</td><td>0.9433</td><td>0.9602</td><td>0.9595</td>
      <td>0.9567</td><td>0.9560</td><td>0.9572</td><td>0.9594</td><td>0.9576</td><td>0.9588</td><td>0.9568</td><td>0.9581</td><td>0.9571</td><td>0.9572</td>
    </tr>
    <tr>
      <td>mBERT</td><td>French</td><td>0.9463</td>
      <td>0.9464</td><td>0.9430</td><td>0.9476</td><td>0.9319</td><td>0.9465</td><td>0.9381</td><td>0.9418</td><td>0.9393</td><td>0.9453</td><td>0.9375</td>
      <td>0.9363</td><td>0.9379</td><td>0.9472</td><td>0.9401</td><td>0.9455</td><td>0.9414</td><td>0.9478</td><td>0.9467</td><td>0.9477</td><td>0.9431</td>
    </tr>
    <tr>
      <td>mBERT</td><td>Hindi</td><td>0.9271</td>
      <td>0.9302</td><td>0.9295</td><td>0.9243</td><td>0.9299</td><td>0.9278</td><td>0.9265</td><td>0.9277</td><td>0.9304</td><td>0.9249</td><td>0.9292</td>
      <td>0.9258</td><td>0.9283</td><td>0.9298</td><td>0.9319</td><td>0.9313</td><td>0.9290</td><td>0.9301</td><td>0.9283</td><td>0.9271</td><td>0.9252</td>
    </tr>
    <tr>
      <td>XLM</td><td>Arabic</td><td>0.9562</td>
      <td>0.9294</td><td>0.9228</td><td>0.9441</td><td>0.9497</td><td>0.9538</td><td>0.9511</td><td>0.9531</td><td>0.9352</td><td>0.9585</td><td>0.9562</td>
      <td>0.9241</td><td>0.9591</td><td>0.9600</td><td>0.9523</td><td>0.9613</td><td>0.9588</td><td>0.9268</td><td>0.9571</td><td>0.9487</td><td>0.9567</td>
    </tr>
    <tr>
      <td>XLM</td><td>Chinese</td><td>0.9636</td>
      <td>0.9665</td><td>0.9641</td><td>0.9659</td><td>0.9638</td><td>0.9638</td><td>0.9659</td><td>0.9655</td><td>0.9666</td><td>0.9650</td><td>0.9660</td>
      <td>0.9661</td><td>0.9661</td><td>0.9676</td><td>0.9638</td><td>0.9644</td><td>0.9648</td><td>0.9653</td><td>0.9655</td><td>0.9661</td><td>0.9635</td>
    </tr>
    <tr>
      <td>XLM</td><td>French</td><td>0.9468</td>
      <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
      <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>XLM</td><td>Hindi</td><td>0.9316</td>
      <td>0.9322</td><td>0.9328</td><td>0.9358</td><td>0.9333</td><td>0.9345</td><td>0.9386</td><td>0.9359</td><td>0.9327</td><td>0.9321</td><td>0.9327</td>
      <td>0.9327</td><td>0.9339</td><td>0.9318</td><td>0.9305</td><td>0.9315</td><td>0.9340</td><td>0.9311</td><td>0.9323</td><td>0.9318</td><td>0.9320</td>
    </tr>
  </tbody>
</table>