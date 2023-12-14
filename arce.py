import math
import torch
import torch.nn as nn

class ApproximateCoprimeFactorization(nn.Module):
    @classmethod
    def sieve_eratosthenes(cls, n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        for p in range(2, int(n**0.5) + 1):
            if is_prime[p]:
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
        primes = [p for p in range(2, n + 1) if is_prime[p]]
        return primes

    @classmethod
    def get_prime_factors(cls, primes, n):
        product = 1
        for prime_idx, prime in enumerate(primes):
            product *= prime
            if product > n:
                return primes[:prime_idx+1]
    
    def approximate_index(self, x, approximation = 1.0):
        return torch.round(x * approximation).int()
    
    def coprime_factorization(self, num_embeddings):
        self.divisors = torch.tensor(self.get_prime_factors(self.sieve_eratosthenes(num_embeddings), num_embeddings))
    
    def quotient_remainder_factorization(self, num_embeddings):
        quotient = math.ceil(math.sqrt(num_embeddings))
        remainder = num_embeddings % quotient
        self.divisors = torch.tensor([quotient, remainder])
        
    def radix_factorization(self, num_embeddings, radix = 10):
        factors = []
        digits_needed = math.ceil(math.log(num_embeddings, radix))
        current_factor = radix
        for i in range(digits_needed):
            factors.append(current_factor)
            current_factor *= radix
        self.divisors = torch.tensor(factors)

    def rotary_only_factorization(self, num_embeddings):
        self.divisors = torch.tensor([num_embeddings])

    def __init__(self, num_embeddings = None, factor_mode = 'radix', approximation = 1.0): #factor_mode must be specified here due to legacy issues with torchfm
        super(ApproximateCoprimeFactorization, self).__init__()
        self.approximation = approximation
        self.num_embeddings = num_embeddings
        self.num_embeddings_approximated = int(round(self.num_embeddings * self.approximation))
        self.factor_mode = factor_mode
        if self.factor_mode == 'coprimes':
            self.coprime_factorization(num_embeddings)
        elif self.factor_mode == 'quotient_remainder':
            self.quotient_remainder_factorization(num_embeddings)
        elif self.factor_mode == 'radix':
            self.radix_factorization(num_embeddings)
        elif self.factor_mode == 'rotary_only':
            self.rotary_only_factorization(num_embeddings)
        else:
            raise ValueError('Unknown factor_mode!')
            
    def forward_wrapper(self, x):
        if self.factor_mode == 'coprimes':
            x = x.unsqueeze(-1) % self.divisors.to(x.device) #(B, I') % (D)
        elif self.factor_mode == 'quotient_remainder':
            x_quotient = x.unsqueeze(-1) // self.divisors[0] #(B, I') // (1)
            x_remainer = x.unsqueeze(-1) % self.divisors[1] #(B, I') % (1)
            x = torch.cat([x_quotient, x_remainer], dim = -1) #(B, I', 2) (D = 2)
        elif self.factor_mode == 'radix':
            x_each_digit = []
            x_remaining = x.clone()
            for divisor in self.divisors:
                x_digit = x_remaining % divisor
                x_remaining = x_remaining // divisor
                x_each_digit.append(x_digit.unsqueeze(-1))
            x = torch.cat(x_each_digit, dim = -1) #(B, I', D)
        elif self.factor_mode == 'rotary_only':
            x = x.unsqueeze(-1)
        return x #(B, I', D)
    
    def apprximation_wrapper(self, x):
        return self.approximate_index(x, self.approximation)

    def forward(self, x):
        x = self.apprximation_wrapper(x)
        x = self.forward_wrapper(x)
        return x

class ApproxRotaryCompEmbed(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, approximation = 0):
        super(ApproxRotaryCompEmbed, self).__init__()
        self.approximation = approximation
        self.embedding_appended = False
        if embedding_dim <= 0:
            raise ValueError('vec_dim must be greater than 0.')
        self.embedding_dim = embedding_dim
        if self.embedding_dim % 2 != 0:
            self.embedding_dim += 1
            self.embedding_appended = True

        if embedding_dim <= 0:
            raise ValueError('embedding_dim must be greater than 0.')
        self.num_embeddings = int(round(num_embeddings * math.exp(self.approximation)))
        if self.num_embeddings < 3:
            self.num_embeddings = 3 # some leniency
        self.factorizer = ApproximateCoprimeFactorization(num_embeddings = self.num_embeddings)
        self.weight = nn.Parameter(torch.randn(1, 1, len(self.factorizer.divisors), 2, self.embedding_dim  // 2))

    def forward(self, x):
        x = self.factorizer(x) # (B, I', D)
        x = x / self.factorizer.divisors.to(x.device) # (B, I', rho) / (D) = (B, I', rho) (rho = D)
        x = x * 2 * math.pi # (B, I', theta) (theta = rho)
        x_cos = torch.cos(x) # (B, I', theta, 2, 2)
        x_sin = torch.sin(x) # (B, I', theta, 2, 2)
        x = x.unsqueeze(-1).unsqueeze(-1) # (B, I', theta, 1, 1)
        x = x.repeat(1, 1, 1, 2, 2) # (B, I', theta, 2, 2)
        x[:, :, :, 0, 0] = x_cos
        x[:, :, :, 0, 1] = -x_sin
        x[:, :, :, 1, 0] = x_sin
        x[:, :, :, 1, 1] = x_cos
        x = x @ self.weight # (B, I', theta, 2, V / 2)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.embedding_dim) # (B, I', theta, V)
        x = x.sum(dim = 2) # (B, I', V)
        if self.embedding_appended:
            x = x[:, :, :-1]
        return x
