# Trading Strategies Using Accounting Data
 
This report summarizes the use of historical financial data as an investment strategy to generate alpha returns. The team adopted nine financial signals to measure three main areas of the firm’s financial condition. These signals help to evaluate the firm’s potential and ability, which would in turn reflect in their stock prices. In addition, the team considered market signals such as Book-To-Market ratio and Market Capitalization in the model. For the accounting signals, each signal is tied to a binary indicator variable where one is assigned when the result is good and zero otherwise. The sum of the nine binary signals would form the F_SCORE. The team created portfolio based on the three-selection criterion: F_SCORE, Market Capitalization and Book-to market ratio.

With reference to Piotroski (2000) and G&C (2013)’s selection criteria, the team considered three strategies: long only portfolio; long/short portfolio; and optimized long/short portfolio with different selection criteria on F_SCORE, market capitalization and Book-to-market ratio.

Among the strategies, Strategy 3: Optimized Long/Short Portfolio produced the highest 20year CAGR of 22.47%, compared to S&P500’s 9.45%. Sharpe Ratio of Strategy 3 (0.87) also outperformed S&P500’s 0.34. The maximum drawdown of 45% compared to S&P500’s 86% also showed that Strategy 3 provided a more stable return.

Our strategies demonstrate that Market Cap, BM ratio and F_SCORE are accurate criteria to select over-valued and financially weak stocks for shorting; and they are very likely to have monotonic relationships with the portfolio return.

In addition, the team recognized that due to the ease of implementation, there are certain limitations to the strategies, such as the use of equal-weighted portfolios. The t-test of difference in annual returns of strategies against benchmark also showed insignificant difference.
