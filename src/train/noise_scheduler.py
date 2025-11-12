import numpy as np
import torch


class LinearScheduler:
    def __init__(self, num_train_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.T = num_train_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
    
    def _compute_betas(self):
        return np.linspace(self.beta_start, self.beta_end, self.T)
    
class CosineScheduler:
    def __init__(self, num_train_steps: int, offset: float = 0.008):
        self.T = num_train_steps
        self.s = offset
        self.alpha_bars = self._compute_alpha_bars()
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas

    def _compute_f(self, time_sample: int):
        return np.cos(((time_sample / self.T + self.s)/(1 + self.s)) * np.pi/ 2) ** 2
    
    def _compute_alpha_bars(self):
        alpha_bars = []
        alpha_bars.append(1.0)
        
        for t in range(1, self.T + 1):
            alpha_bars.append(self._compute_f(t) / self._compute_f(0))
        return np.array(alpha_bars)
    
    def _compute_betas(self):
        betas = []
        for t in range(1, self.T + 1):
            beta_t = 1 - self.alpha_bars[t] / self.alpha_bars[t - 1]
            betas.append(beta_t)
        return np.array(betas)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    scheduler = CosineScheduler(num_train_steps=10)
    print("Betas:", scheduler.betas)
    print("Alpha bars:", scheduler.alpha_bars)
    plt.plot(np.arange(1, 11), scheduler.betas)
    plt.title("Betas")
    plt.xlabel("Training Steps")
    plt.ylabel("Beta Value")
    plt.grid()
    plt.show()

    plt.plot(np.arange(0, 11), scheduler.alpha_bars)
    plt.title("Alpha Bars")
    plt.xlabel("Training Steps")
    plt.ylabel("Alpha Bar Value")
    plt.grid()
    plt.show()