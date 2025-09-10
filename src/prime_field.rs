use crate::matrix::{Identity, MatrixElement, Zero};

use std::cmp::PartialEq;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

#[derive(Copy, Clone, Debug)]
pub struct PrimeFieldElement {
    val: u64,
    p: u64,
}

pub fn modular_inverse_prime(a: u64, m: u64) -> u64 {
    // The inverse of 0 is not defined.
    assert!(a != 0, "Cannot find the modular inverse of 0.");
    assert!(m > 1, "Modulus must be greater than 1.");

    let mut old_r: i64 = a as i64;
    let mut r: i64 = m as i64;
    let mut old_s: i64 = 1;
    let mut s: i64 = 0;

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }

    let m_i = m as i64;
    ((old_s % m_i + m_i) % m_i) as u64
}

impl PrimeFieldElement {
    pub fn new(val: u64, p: u64) -> Self {
        assert!(p > 1, "Modulus must be greater than 1.");
        PrimeFieldElement { val: val % p, p }
    }

    pub fn modulus(&self) -> u64 {
        self.p
    }
    pub fn value(&self) -> u64 {
        self.val
    }

    pub fn pow(&self, k: i64) -> Self {
        if k == 0 {
            return self.identity();
        }

        if k < 0 {
            // a^{-k} = (a^{-1})^{k}
            return (self.identity() / *self).pow(-k);
        }

        if k == 1 {
            return *self;
        }

        let half = self.pow(k / 2);
        let mut ans = half * half;
        if k % 2 != 0 {
            ans = ans * *self;
        }
        ans
    }

    pub fn identity(&self) -> Self {
        PrimeFieldElement { val: 1, p: self.p }
    }

    pub fn zero(&self) -> Self {
        PrimeFieldElement { val: 0, p: self.p }
    }
}

impl Add for PrimeFieldElement {
    type Output = Self;

    fn add(self, rhs: PrimeFieldElement) -> Self {
        assert!(
            self.p == rhs.p,
            "Mismatched moduli in addition: {} vs {}",
            self.p,
            rhs.p
        );
        PrimeFieldElement {
            val: (self.val + rhs.val) % self.p,
            p: self.p,
        }
    }
}

impl Sub for PrimeFieldElement {
    type Output = Self;

    fn sub(self, rhs: PrimeFieldElement) -> Self {
        assert!(
            self.p == rhs.p,
            "Mismatched moduli in subtraction: {} vs {}",
            self.p,
            rhs.p
        );
        PrimeFieldElement {
            val: (self.val + (self.p - rhs.val)) % self.p,
            p: self.p,
        }
    }
}

impl Mul for PrimeFieldElement {
    type Output = Self;

    fn mul(self, rhs: PrimeFieldElement) -> Self {
        assert!(
            self.p == rhs.p,
            "Mismatched moduli in multiplication: {} vs {}",
            self.p,
            rhs.p
        );
        PrimeFieldElement {
            val: (self.val.wrapping_mul(rhs.val)) % self.p,
            p: self.p,
        }
    }
}

impl Div for PrimeFieldElement {
    type Output = Self;

    fn div(self, rhs: PrimeFieldElement) -> Self {
        assert!(
            self.p == rhs.p,
            "Mismatched moduli in division: {} vs {}",
            self.p,
            rhs.p
        );
        assert!(
            rhs.val != 0,
            "Cannot divide by 0. LHS = {:?}, RHS = {:?}",
            self,
            rhs
        );
        PrimeFieldElement {
            val: (self.val * modular_inverse_prime(rhs.val, self.p)) % self.p,
            p: self.p,
        }
    }
}

impl Neg for PrimeFieldElement {
    type Output = Self;

    fn neg(self) -> Self {
        if self.val == 0 {
            return self;
        }
        PrimeFieldElement {
            val: self.p - self.val,
            p: self.p,
        }
    }
}

impl PartialEq for PrimeFieldElement {
    fn eq(&self, rhs: &PrimeFieldElement) -> bool {
        assert!(
            self.p == rhs.p,
            "Mismatched moduli in subtraction: {} vs {}",
            self.p,
            rhs.p
        );
        self.val == rhs.val
    }

    fn ne(&self, rhs: &PrimeFieldElement) -> bool {
        assert!(
            self.p == rhs.p,
            "Mismatched moduli in subtraction: {} vs {}",
            self.p,
            rhs.p
        );
        self.val != rhs.val
    }
}

impl AddAssign for PrimeFieldElement {
    fn add_assign(&mut self, rhs: Self) {
        assert!(
            self.p == rhs.p,
            "Mismatched moduli in add_assign: {} vs {}",
            self.p,
            rhs.p
        );
        self.val = (self.val + rhs.val) % self.p;
    }
}

impl Zero for PrimeFieldElement {
    fn zero() -> Self {
        panic!(
            "PrimeFieldElement::zero() should not be called. Use identity(&self) and zero(&self) methods instead."
        );
    }
}

impl Identity for PrimeFieldElement {
    fn identity() -> Self {
        panic!(
            "PrimeFieldElement::identity() should not be called. Use identity(&self) and zero(&self) methods instead."
        );
    }
}

impl MatrixElement for PrimeFieldElement {}

#[cfg(test)]
mod tests {
    use super::*;

    type GF = PrimeFieldElement;

    #[test]
    fn test_modular_inverse_prime_basic() {
        assert_eq!(modular_inverse_prime(1, 7), 1);
        assert_eq!(modular_inverse_prime(2, 7), 4); // 2*4 = 8 ≡ 1 (mod 7)
        assert_eq!(modular_inverse_prime(3, 7), 5); // 3*5 = 15 ≡ 1 (mod 7)
        assert_eq!(modular_inverse_prime(5, 7), 3); // 5*3 = 15 ≡ 1 (mod 7)
        assert_eq!(modular_inverse_prime(6, 7), 6); // self-inverse in GF(7)
    }

    #[test]
    fn test_new_and_eq() {
        let p = 7;
        // 10 mod 7 = 3
        assert_eq!(GF::new(10, p), GF::new(3, p));
        assert_ne!(GF::new(10, p), GF::new(4, p));
    }

    #[test]
    fn test_addition() {
        let p = 7;
        let a = GF::new(3, p);
        let b = GF::new(6, p);
        assert_eq!(a + b, GF::new(2, p)); // 3 + 6 = 9 ≡ 2 (mod 7)
    }

    #[test]
    fn test_add_assign() {
        let p = 7;
        let mut a = GF::new(3, p);
        a += GF::new(6, p);
        assert_eq!(a, GF::new(2, p));
    }

    #[test]
    fn test_subtraction() {
        let p = 7;
        let a = GF::new(3, p);
        let b = GF::new(6, p);
        assert_eq!(a - b, GF::new(4, p)); // 3 - 6 = -3 ≡ 4 (mod 7)
    }

    #[test]
    fn test_multiplication() {
        let p = 7;
        let a = GF::new(3, p);
        let b = GF::new(5, p);
        assert_eq!(a * b, GF::new(1, p)); // 15 ≡ 1 (mod 7)
    }

    #[test]
    fn test_negation() {
        let p = 7;
        let a = GF::new(3, p);
        assert_eq!(-a, GF::new(4, p)); // -3 ≡ 4 (mod 7)
        assert_eq!(a + (-a), a.zero());
    }

    #[test]
    fn test_zero_and_identity() {
        let p = 7;
        let a = GF::new(5, p);
        assert_eq!(a.zero() + a, a);
        assert_eq!(a + a.zero(), a);
        assert_eq!(a.identity() * a, a);
        assert_eq!(a * a.identity(), a);
    }

    #[test]
    fn test_division() {
        let p = 7;
        let a = GF::new(5, p);
        let b = GF::new(3, p);
        // a / b = a * b^{-1}; in GF(7), 3^{-1} = 5, so result = 5*5 = 25 ≡ 4
        assert_eq!(a / b, GF::new(4, p));
        // sanity: (a / b) * b = a
        assert_eq!((a / b) * b, a);
    }

    #[test]
    fn test_pow_zero() {
        let p = 7;
        let a = GF::new(5, p);
        assert_eq!(a.pow(0), a.identity());
    }

    #[test]
    fn test_pow_positive() {
        let p = 7;
        let a = GF::new(3, p);
        // 3^2 = 9 ≡ 2; 3^3 = 27 ≡ 6; 3^4 = 81 ≡ 4 (mod 7)
        assert_eq!(a.pow(2), GF::new(2, p));
        assert_eq!(a.pow(3), GF::new(6, p));
        assert_eq!(a.pow(4), GF::new(4, p));
    }

    #[test]
    fn test_pow_negative() {
        let p = 7;
        let a = GF::new(3, p);
        // 3^{-1} ≡ 5 (mod 7)
        assert_eq!(a.pow(-1), GF::new(5, p));
        // 3^{-2} ≡ (3^{-1})^2 ≡ 5^2 = 25 ≡ 4
        assert_eq!(a.pow(-2), GF::new(4, p));
    }

    #[test]
    fn test_fermats_little_theorem() {
        let p = 7;
        // For non-zero a in GF(p), a^{p-1} = 1
        let a = GF::new(2, p);
        assert_eq!(a.pow(6), a.identity());
    }
}
