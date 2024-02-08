import math
import torch
import torch.nn as nn
from functools import reduce

class ApproximateCoprimeFactorization(nn.Module):
    
    @staticmethod
    def sieve_eratosthenes(n: int) -> list:
        """Fast generation of prime numbers using sieve of Eratosthenes.

        Args:
            n (int): the upper bound of the prime numbers to be generated.

        Returns:
            list: the list of prime numbers less than or equal to n.
        """
        precomputed_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        precomputed_products = [reduce(lambda x, y: x*y, precomputed_primes[:i]) for i in range(1, len(precomputed_primes)+1)]
        if n < precomputed_products[-1]:
            return precomputed_primes
        else:
            primes = precomputed_primes
            for i in range(precomputed_primes[-1]+1, n):
                is_prime = True
                for prime in primes:
                    if i % prime == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(i)
                    precomputed_products.append(precomputed_products[-1] * i)
                    if precomputed_products[-1] > n:
                        return primes

    @staticmethod
    def get_prime_factors(primes: list, n: int) -> list:
        """Get the prime factors of n.

        Args:
            primes (list): the list of prime numbers less than or equal to n.
            n (int): the number to be factorized.
        Returns:
            list: the list of prime factors of n.
        """        
        product = 1
        for prime_idx, prime in enumerate(primes):
            product *= prime
            if product > n:
                return primes[:prime_idx+1]
    
    def approximate_index(self: 'ApproximateCoprimeFactorization', x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        """Tunable approximation of the index.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            x (torch.Tensor): the input tensor of indices.
            dilation (int, optional): the dilation factor. Defaults to 1.

        Returns:
            torch.Tensor: the approximated indices.
        """        
        rand_shift = torch.randint(0, dilation, x.shape, device = x.device)
        return x * dilation + rand_shift
    
    def coprime_factorization(self: 'ApproximateCoprimeFactorization', num_embeddings: int) -> None:
        """Intialize factors for the coprime factorization.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            num_embeddings (int): the number of embeddings.
        """        
        self.divisors = torch.tensor(self.get_prime_factors(self.sieve_eratosthenes(num_embeddings), num_embeddings))
    
    def quotient_remainder_factorization(self: 'ApproximateCoprimeFactorization', num_embeddings: int) -> None:
        """Intialize factors for the quotient-remainder factorization.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            num_embeddings (int): the number of embeddings.
        """        
        quotient = math.ceil(math.sqrt(num_embeddings))
        remainder = num_embeddings % quotient
        self.divisors = torch.tensor([quotient, remainder])
        
    def radix_factorization(self: 'ApproximateCoprimeFactorization', num_embeddings: int, radix: int = 10) -> None:
        """Intialize factors for the radix factorization.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            num_embeddings (int): the number of embeddings.
            radix (int, optional): the radix. Defaults to 10.
        """
        factors = []
        digits_needed = math.ceil(math.log(num_embeddings, radix))
        current_factor = radix
        for _ in range(digits_needed):
            factors.append(current_factor)
            current_factor *= radix
        self.divisors = torch.tensor(factors)

    def rotary_only_factorization(self: 'ApproximateCoprimeFactorization', num_embeddings: int) -> None:
        """Intialize factors for the rotary-only factorization.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            num_embeddings (int): the number of embeddings.
        """
        self.divisors = torch.tensor([num_embeddings])

    def __init__(self: 'ApproximateCoprimeFactorization', num_embeddings: int, factor_mode: str = 'coprimes', approximation: float = 1.0, **kwargs) -> None:
        """Initialize the class.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            num_embeddings (int): the number of embeddings.
            factor_mode (str, optional): the factorization mode. Defaults to 'rotary_only'.
            approximation (float, optional): the approximation factor. Defaults to 1.0 (maximum approximation).

        Raises:
            ValueError: if the factor_mode is not supported. Available options are 'coprimes', 'quotient_remainder', 'radix', and 'rotary_only'.
        """        
        super(ApproximateCoprimeFactorization, self).__init__()
        self.approximation = approximation
        if self.approximation < 1e-8 or self.approximation > 1:
            raise ValueError('Approximation factor must be between 1e-8 and 1.')
        self.dilation = int(round(1 / self.approximation))
        self.num_embeddings = (1 + num_embeddings) * self.dilation
        self.factor_mode = factor_mode
        if self.factor_mode == 'coprimes':
            self.coprime_factorization(self.num_embeddings)
        elif self.factor_mode == 'quotient_remainder':
            self.quotient_remainder_factorization(self.num_embeddings)
        elif self.factor_mode == 'radix':
            self.radix_factorization(self.num_embeddings, radix=kwargs.get('radix', 10))
        elif self.factor_mode == 'rotary_only':
            self.rotary_only_factorization(self.num_embeddings)
        else:
            raise ValueError('Unknown factor_mode!')
            
    def forward_wrapper(self: 'ApproximateCoprimeFactorization', x: torch.Tensor) -> torch.Tensor:
        """Forward wrapper for the factorization.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            x (torch.Tensor): the input tensor of indices.

        Returns:
            torch.Tensor: the factorized indices.
        """
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
    
    def approximation_wrapper(self: 'ApproximateCoprimeFactorization', x: torch.Tensor) -> torch.Tensor:
        """Approximation wrapper for the factorization.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            x (torch.Tensor): the input tensor of indices.

        Returns:
            torch.Tensor: the approximated indices.
        """
        return self.approximate_index(x, dilation = self.dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            self (ApproximateCoprimeFactorization): the instance of the class.
            x (torch.Tensor): the input tensor of indices.

        Returns:
            torch.Tensor: the approximated and factorized indices.
        """
        x = self.approximation_wrapper(x)
        x = self.forward_wrapper(x)
        return x

class ApproxRotaryCompEmbed(nn.Module):
    def __init__(self: 'ApproxRotaryCompEmbed', num_embeddings: int, embedding_dim: int, factor_mode: str = 'coprimes', approximation: float = 1.0,) -> None:
        """Initialize the class.

        Args:
            self (ApproxRotaryCompEmbed): the instance of the class.
            num_embeddings (int): the number of embeddings.
            embedding_dim (int): the dimension of the embeddings.
            approximation (float, optional): the approximation factor. Defaults to 1.0.

        Raises:
            ValueError: if the vec_dim is non-positive.
            ValueError: if the num_embeddings is non-positive.
        """        
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
        self.factorizer = ApproximateCoprimeFactorization(num_embeddings = self.num_embeddings, factor_mode = factor_mode, approximation = self.approximation)
        self.weight = nn.Parameter(torch.randn(1, 1, len(self.factorizer.divisors), 2, self.embedding_dim  // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            self (ApproxRotaryCompEmbed): the instance of the class.
            x (torch.Tensor): the input tensor of indices.

        Returns:
            torch.Tensor: the embeddings.
        """
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