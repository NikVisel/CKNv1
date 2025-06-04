//! LaTeX parser for converting mathematical expressions to trees

use crate::algebra::{Tree, TreeBuilder};
use crate::Result;

/// Types of mathematical expressions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprType {
    /// Number or variable
    Atom(String),
    /// Binary operation (+, -, *, /, ^)
    BinOp {
        op: String,
        left: Box<MathExpr>,
        right: Box<MathExpr>,
    },
    /// Unary operation (-, sqrt, etc.)
    UnaryOp {
        op: String,
        arg: Box<MathExpr>,
    },
    /// Function call (sin, cos, log, etc.)
    Function {
        name: String,
        args: Vec<MathExpr>,
    },
    /// Fraction
    Fraction {
        num: Box<MathExpr>,
        den: Box<MathExpr>,
    },
    /// Subscript/Superscript
    Script {
        base: Box<MathExpr>,
        sub: Option<Box<MathExpr>>,
        sup: Option<Box<MathExpr>>,
    },
    /// Parentheses group
    Group(Box<MathExpr>),
    /// Sum/Product with limits
    BigOp {
        op: String,
        lower: Option<Box<MathExpr>>,
        upper: Option<Box<MathExpr>>,
        body: Box<MathExpr>,
    },
}

/// Mathematical expression
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathExpr {
    pub expr_type: ExprType,
}

impl MathExpr {
    /// Create atomic expression
    pub fn atom(s: &str) -> Self {
        MathExpr {
            expr_type: ExprType::Atom(s.to_string()),
        }
    }
    
    /// Create binary operation
    pub fn binop(op: &str, left: MathExpr, right: MathExpr) -> Self {
        MathExpr {
            expr_type: ExprType::BinOp {
                op: op.to_string(),
                left: Box::new(left),
                right: Box::new(right),
            },
        }
    }
    
    /// Create fraction
    pub fn fraction(num: MathExpr, den: MathExpr) -> Self {
        MathExpr {
            expr_type: ExprType::Fraction {
                num: Box::new(num),
                den: Box::new(den),
            },
        }
    }
    
    /// Convert to tree representation
    pub fn to_tree(&self) -> crate::Result<Tree> {
        let mut builder = TreeBuilder::new();
        let mut node_counter = 1;
        
        self.build_tree_recursive(&mut builder, 0, &mut node_counter)?;
        
        builder.build()
    }
    
    /// Recursively build tree
    fn build_tree_recursive(
        &self,
        builder: &mut TreeBuilder,
        parent: usize,
        counter: &mut usize,
    ) -> crate::Result<()> {
        match &self.expr_type {
            ExprType::Atom(_) => {
                // Leaf node - no children
            }
            ExprType::BinOp { left, right, .. } => {
                // Add left child
                let left_node = *counter;
                builder.add_child(parent, left_node);
                *counter += 1;
                left.build_tree_recursive(builder, left_node, counter)?;
                
                // Add right child
                let right_node = *counter;
                builder.add_child(parent, right_node);
                *counter += 1;
                right.build_tree_recursive(builder, right_node, counter)?;
            }
            ExprType::UnaryOp { arg, .. } => {
                // Add single child
                let child_node = *counter;
                builder.add_child(parent, child_node);
                *counter += 1;
                arg.build_tree_recursive(builder, child_node, counter)?;
            }
            ExprType::Function { args, .. } => {
                // Add child for each argument
                for arg in args {
                    let child_node = *counter;
                    builder.add_child(parent, child_node);
                    *counter += 1;
                    arg.build_tree_recursive(builder, child_node, counter)?;
                }
            }
            ExprType::Fraction { num, den } => {
                // Add numerator
                let num_node = *counter;
                builder.add_child(parent, num_node);
                *counter += 1;
                num.build_tree_recursive(builder, num_node, counter)?;
                
                // Add denominator
                let den_node = *counter;
                builder.add_child(parent, den_node);
                *counter += 1;
                den.build_tree_recursive(builder, den_node, counter)?;
            }
            ExprType::Script { base, sub, sup } => {
                // Add base
                let base_node = *counter;
                builder.add_child(parent, base_node);
                *counter += 1;
                base.build_tree_recursive(builder, base_node, counter)?;
                
                // Add subscript if present
                if let Some(sub_expr) = sub {
                    let sub_node = *counter;
                    builder.add_child(parent, sub_node);
                    *counter += 1;
                    sub_expr.build_tree_recursive(builder, sub_node, counter)?;
                }
                
                // Add superscript if present
                if let Some(sup_expr) = sup {
                    let sup_node = *counter;
                    builder.add_child(parent, sup_node);
                    *counter += 1;
                    sup_expr.build_tree_recursive(builder, sup_node, counter)?;
                }
            }
            ExprType::Group(inner) => {
                // Transparent - just process inner expression
                inner.build_tree_recursive(builder, parent, counter)?;
            }
            ExprType::BigOp { body, lower, upper, .. } => {
                // Add lower limit if present
                if let Some(lower_expr) = lower {
                    let lower_node = *counter;
                    builder.add_child(parent, lower_node);
                    *counter += 1;
                    lower_expr.build_tree_recursive(builder, lower_node, counter)?;
                }
                
                // Add upper limit if present
                if let Some(upper_expr) = upper {
                    let upper_node = *counter;
                    builder.add_child(parent, upper_node);
                    *counter += 1;
                    upper_expr.build_tree_recursive(builder, upper_node, counter)?;
                }
                
                // Add body
                let body_node = *counter;
                builder.add_child(parent, body_node);
                *counter += 1;
                body.build_tree_recursive(builder, body_node, counter)?;
            }
        }
        
        Ok(())
    }
}

/// Simple LaTeX parser
pub struct LaTeXParser {
    /// Current position in input
    pos: usize,
    /// Input string
    input: String,
}

impl LaTeXParser {
    /// Create new parser
    pub fn new(input: &str) -> Self {
        LaTeXParser {
            pos: 0,
            input: input.to_string(),
        }
    }
    
    /// Parse LaTeX expression
    pub fn parse(&mut self) -> crate::Result<MathExpr> {
        self.skip_whitespace();
        self.parse_expression()
    }
    
    /// Parse expression (handles precedence)
    fn parse_expression(&mut self) -> crate::Result<MathExpr> {
        // Start with addition/subtraction (lowest precedence)
        self.parse_additive()
    }
    
    /// Parse additive expression (+ -)
    fn parse_additive(&mut self) -> crate::Result<MathExpr> {
        let mut left = self.parse_multiplicative()?;
        
        while self.pos < self.input.len() {
            self.skip_whitespace();
            if self.peek_char() == Some('+') {
                self.consume_char();
                let right = self.parse_multiplicative()?;
                left = MathExpr::binop("+", left, right);
            } else if self.peek_char() == Some('-') {
                self.consume_char();
                let right = self.parse_multiplicative()?;
                left = MathExpr::binop("-", left, right);
            } else {
                break;
            }
        }
        
        Ok(left)
    }
    
    /// Parse multiplicative expression (* / \cdot \times)
    fn parse_multiplicative(&mut self) -> crate::Result<MathExpr> {
        let mut left = self.parse_power()?;
        
        while self.pos < self.input.len() {
            self.skip_whitespace();
            if self.peek_char() == Some('*') {
                self.consume_char();
                let right = self.parse_power()?;
                left = MathExpr::binop("*", left, right);
            } else if self.peek_char() == Some('/') {
                self.consume_char();
                let right = self.parse_power()?;
                left = MathExpr::binop("/", left, right);
            } else if self.peek_str("\\cdot") {
                self.consume_str("\\cdot");
                let right = self.parse_power()?;
                left = MathExpr::binop("*", left, right);
            } else if self.peek_str("\\times") {
                self.consume_str("\\times");
                let right = self.parse_power()?;
                left = MathExpr::binop("*", left, right);
            } else {
                break;
            }
        }
        
        Ok(left)
    }
    
    /// Parse power expression (^)
    fn parse_power(&mut self) -> crate::Result<MathExpr> {
        let mut base = self.parse_primary()?;
        
        self.skip_whitespace();
        if self.peek_char() == Some('^') {
            self.consume_char();
            let exp = self.parse_primary()?;
            base = MathExpr::binop("^", base, exp);
        }
        
        Ok(base)
    }
    
    /// Parse primary expression (atoms, fractions, functions, groups)
    fn parse_primary(&mut self) -> crate::Result<MathExpr> {
        self.skip_whitespace();
        
        // Check for LaTeX commands
        if self.peek_char() == Some('\\') {
            if self.peek_str("\\frac") {
                return self.parse_fraction();
            } else if self.peek_str("\\sum") {
                return self.parse_big_op("sum");
            } else if self.peek_str("\\prod") {
                return self.parse_big_op("prod");
            } else {
                // Try to parse function
                return self.parse_function();
            }
        }
        
        // Check for parentheses
        if self.peek_char() == Some('(') || self.peek_char() == Some('{') {
            return self.parse_group();
        }
        
        // Otherwise parse atom
        self.parse_atom()
    }
    
    /// Parse fraction \frac{num}{den}
    fn parse_fraction(&mut self) -> crate::Result<MathExpr> {
        self.consume_str("\\frac");
        self.skip_whitespace();
        
        // Parse numerator
        self.expect_char('{')?;
        let num = self.parse_expression()?;
        self.expect_char('}')?;
        
        // Parse denominator
        self.skip_whitespace();
        self.expect_char('{')?;
        let den = self.parse_expression()?;
        self.expect_char('}')?;
        
        Ok(MathExpr::fraction(num, den))
    }
    
    /// Parse big operator with limits
    fn parse_big_op(&mut self, op: &str) -> crate::Result<MathExpr> {
        self.consume_str(&format!("\\{}", op));
        self.skip_whitespace();
        
        let mut lower = None;
        let mut upper = None;
        
        // Check for limits
        if self.peek_char() == Some('_') {
            self.consume_char();
            lower = Some(Box::new(self.parse_primary()?));
        }
        
        self.skip_whitespace();
        if self.peek_char() == Some('^') {
            self.consume_char();
            upper = Some(Box::new(self.parse_primary()?));
        }
        
        // Parse body (usually the expression being summed/multiplied)
        self.skip_whitespace();
        let body = Box::new(self.parse_primary()?);
        
        Ok(MathExpr {
            expr_type: ExprType::BigOp {
                op: op.to_string(),
                lower,
                upper,
                body,
            },
        })
    }
    
    /// Parse function call
    fn parse_function(&mut self) -> crate::Result<MathExpr> {
        self.consume_char(); // consume '\'
        let name = self.parse_identifier();
        
        self.skip_whitespace();
        let arg = self.parse_primary()?;
        
        Ok(MathExpr {
            expr_type: ExprType::Function {
                name,
                args: vec![arg],
            },
        })
    }
    
    /// Parse parentheses group
    fn parse_group(&mut self) -> crate::Result<MathExpr> {
        let open = self.consume_char();
        let close = match open {
            '(' => ')',
            '{' => '}',
            '[' => ']',
            _ => return Err(crate::HopfMLError::AlgebraError("Invalid grouping".to_string())),
        };
        
        let inner = self.parse_expression()?;
        self.expect_char(close)?;
        
        Ok(MathExpr {
            expr_type: ExprType::Group(Box::new(inner)),
        })
    }
    
    /// Parse atom (number or variable)
    fn parse_atom(&mut self) -> crate::Result<MathExpr> {
        self.skip_whitespace();
        
        // Parse number
        if self.peek_char().map_or(false, |c| c.is_ascii_digit()) {
            let num = self.parse_number();
            return Ok(MathExpr::atom(&num));
        }
        
        // Parse variable
        if self.peek_char().map_or(false, |c| c.is_alphabetic()) {
            let var = self.consume_char().to_string();
            
            // Check for subscript/superscript
            let mut sub = None;
            let mut sup = None;
            
            if self.peek_char() == Some('_') {
                self.consume_char();
                sub = Some(Box::new(self.parse_primary()?));
            }
            
            if self.peek_char() == Some('^') {
                self.consume_char();
                sup = Some(Box::new(self.parse_primary()?));
            }
            
            if sub.is_some() || sup.is_some() {
                return Ok(MathExpr {
                    expr_type: ExprType::Script {
                        base: Box::new(MathExpr::atom(&var)),
                        sub,
                        sup,
                    },
                });
            }
            
            return Ok(MathExpr::atom(&var));
        }
        
        Err(crate::HopfMLError::AlgebraError("Expected atom".to_string()))
    }
    
    // Helper methods
    
    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input.chars().nth(self.pos).unwrap().is_whitespace() {
            self.pos += 1;
        }
    }
    
    fn peek_char(&self) -> Option<char> {
        self.input.chars().nth(self.pos)
    }
    
    fn peek_str(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }
    
    fn consume_char(&mut self) -> char {
        let ch = self.input.chars().nth(self.pos).unwrap();
        self.pos += 1;
        ch
    }
    
    fn consume_str(&mut self, s: &str) {
        self.pos += s.len();
    }
    
    fn expect_char(&mut self, ch: char) -> crate::Result<()> {
        self.skip_whitespace();
        if self.peek_char() == Some(ch) {
            self.consume_char();
            Ok(())
        } else {
            Err(crate::HopfMLError::AlgebraError(format!("Expected '{}'", ch)))
        }
    }
    
    fn parse_identifier(&mut self) -> String {
        let mut ident = String::new();
        while self.pos < self.input.len() {
            if let Some(ch) = self.peek_char() {
                if ch.is_alphabetic() {
                    ident.push(self.consume_char());
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        ident
    }
    
    fn parse_number(&mut self) -> String {
        let mut num = String::new();
        while self.pos < self.input.len() {
            if let Some(ch) = self.peek_char() {
                if ch.is_ascii_digit() || ch == '.' {
                    num.push(self.consume_char());
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        num
    }
}

/// Parse LaTeX to tree
pub fn latex_to_tree(latex: &str) -> crate::Result<Tree> {
    let mut parser = LaTeXParser::new(latex);
    let expr = parser.parse()?;
    expr.to_tree()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_expr() {
        let expr = MathExpr::binop("+", 
            MathExpr::atom("x"),
            MathExpr::atom("y")
        );
        
        let tree = expr.to_tree().unwrap();
        assert_eq!(tree.size(), 3); // root + 2 leaves
    }
    
    #[test]
    fn test_fraction() {
        let expr = MathExpr::fraction(
            MathExpr::atom("a"),
            MathExpr::binop("+", MathExpr::atom("b"), MathExpr::atom("c"))
        );
        
        let tree = expr.to_tree().unwrap();
        assert_eq!(tree.size(), 5);
    }
    
    #[test]
    fn test_latex_parser() {
        let latex = "x + y";
        let tree = latex_to_tree(latex).unwrap();
        assert_eq!(tree.size(), 3);
        
        let latex2 = "\\frac{a}{b + c}";
        let tree2 = latex_to_tree(latex2).unwrap();
        assert_eq!(tree2.size(), 5);
    }
} 