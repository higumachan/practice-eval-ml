use anyhow::bail;
use itertools::Itertools;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub enum Value {
    Int(i32),
    Bool(bool),
    Error,
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Error => write!(f, "error"),
        }
    }
}

impl Value {
    pub fn int(&self) -> Option<i32> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Operator2 {
    Add,
    Sub,
    Mul,
    Lt,
}

impl Operator2 {
    pub fn eval(&self, left: &Value, right: &Value) -> Result<Value, ApplyError> {
        let left = left.int().ok_or_else(|| {
            ApplyError::RuntimeError(left.clone(), LeftOrRight::Left, self.clone())
        })?;
        let right = right.int().ok_or_else(|| {
            ApplyError::RuntimeError(right.clone(), LeftOrRight::Right, self.clone())
        })?;
        match self {
            Operator2::Add => Ok(Value::Int(left + right)),
            Operator2::Sub => Ok(Value::Int(left - right)),
            Operator2::Mul => Ok(Value::Int(left * right)),
            Operator2::Lt => Ok(Value::Bool(left < right)),
        }
    }

    pub fn to_alpha_string(&self) -> &'static str {
        match self {
            Operator2::Add => "plus",
            Operator2::Sub => "minus",
            Operator2::Mul => "times",
            Operator2::Lt => "less than",
        }
    }

    pub fn to_expr_rule(&self) -> Option<Rule> {
        match self {
            Operator2::Add => Some(Rule::EPlus),
            Operator2::Sub => Some(Rule::EMinus),
            Operator2::Mul => Some(Rule::ETimes),
            Operator2::Lt => Some(Rule::ELt),
            _ => None,
        }
    }

    pub fn to_binary_rule(&self) -> Option<Rule> {
        match self {
            Operator2::Add => Some(Rule::BPlus),
            Operator2::Sub => Some(Rule::BMinus),
            Operator2::Mul => Some(Rule::BTimes),
            Operator2::Lt => Some(Rule::BLt),
            _ => None,
        }
    }
}

impl Display for Operator2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator2::Add => write!(f, "+"),
            Operator2::Sub => write!(f, "-"),
            Operator2::Mul => write!(f, "*"),
            Operator2::Lt => write!(f, "<"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i32),
    Bool(bool),
    Variable(Ident),
    Operand2(Box<Expr>, Operator2, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Let(Ident, Box<Expr>, Box<Expr>),
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Int(i) => write!(f, "{}", i),
            Expr::Bool(b) => write!(f, "{}", b),
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Operand2(e1, op, e2) => write!(f, "({} {} {})", e1, op, e2),
            Expr::If(e1, e2, e3) => write!(f, "(if {} then {} else {})", e1, e2, e3),
            Expr::Let(s, e1, e2) => write!(f, "(let {} = {} in {})", s, e1, e2),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Rule {
    EInt,
    EBool,
    EPlus,
    EMinus,
    ETimes,
    ELt,
    EIfT,
    EIfF,
    BPlus,
    BMinus,
    BTimes,
    BLt,
    EIfInt,
    EPlusBoolL,
    EPlusBoolR,
    EMinusBoolL,
    EMinusBoolR,
    ETimesBoolL,
    ETimesBoolR,
    ELtBoolL,
    ELtBoolR,
    EIfError,
    EIfTError,
    EIfFError,
    EPlusErrorL,
    EPlusErrorR,
    EMinusErrorL,
    EMinusErrorR,
    ETimesErrorL,
    ETimesErrorR,
    ELtErrorL,
    ELtErrorR,
}

impl Display for Rule {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Rule::EInt => write!(f, "E-Int"),
            Rule::EBool => write!(f, "E-Bool"),
            Rule::EPlus => write!(f, "E-Plus"),
            Rule::BPlus => write!(f, "B-Plus"),
            Rule::EMinus => write!(f, "E-Minus"),
            Rule::BMinus => write!(f, "B-Minus"),
            Rule::ETimes => write!(f, "E-Times"),
            Rule::BTimes => write!(f, "B-Times"),
            Rule::EIfT => write!(f, "E-IfT"),
            Rule::EIfF => write!(f, "E-IfF"),
            Rule::ELt => write!(f, "E-Lt"),
            Rule::BLt => write!(f, "B-Lt"),
            Rule::EIfInt => write!(f, "E-IfInt"),
            Rule::EPlusBoolL => write!(f, "E-PlusBoolL"),
            Rule::EPlusBoolR => write!(f, "E-PlusBoolR"),
            Rule::EMinusBoolL => write!(f, "E-MinusBoolL"),
            Rule::EMinusBoolR => write!(f, "E-MinusBoolR"),
            Rule::ETimesBoolL => write!(f, "E-TimesBoolL"),
            Rule::ETimesBoolR => write!(f, "E-TimesBoolR"),
            Rule::ELtBoolL => write!(f, "E-LtBoolL"),
            Rule::ELtBoolR => write!(f, "E-LtBoolR"),
            Rule::EIfError => write!(f, "E-IfError"),
            Rule::EIfTError => write!(f, "E-IfTError"),
            Rule::EIfFError => write!(f, "E-IfFError"),
            Rule::EPlusErrorL => write!(f, "E-PlusErrorL"),
            Rule::EPlusErrorR => write!(f, "E-PlusErrorR"),
            Rule::EMinusErrorL => write!(f, "E-MinusErrorL"),
            Rule::EMinusErrorR => write!(f, "E-MinusErrorR"),
            Rule::ETimesErrorL => write!(f, "E-TimesErrorL"),
            Rule::ETimesErrorR => write!(f, "E-TimesErrorR"),
            Rule::ELtErrorL => write!(f, "E-LtErrorL"),
            Rule::ELtErrorR => write!(f, "E-LtErrorR"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExprNode {
    expr: Expr,
    eval_to: Value,
    rule: Rule,
    children: Vec<Box<DerivationNode>>,
}

impl ExprNode {
    pub fn new(expr: Expr, eval_to: Value, rule: Rule, children: Vec<Box<DerivationNode>>) -> Self {
        Self {
            expr,
            eval_to,
            rule,
            children,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DerivationNode {
    Expr(ExprNode),
    Error {
        expr: Expr,
        rule: Rule,
        children: Vec<Box<DerivationNode>>,
    },
    Binary {
        left: Value,
        operand: &'static str,
        right: Value,
        answer: Value,
        rule: Rule,
    },
}

impl DerivationNode {
    pub fn eval_to(&self) -> Option<&Value> {
        match self {
            DerivationNode::Expr(expr_node) => Some(&expr_node.eval_to),
            DerivationNode::Error { .. } => Some(&Value::Error),
            DerivationNode::Binary { answer, .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LeftOrRight {
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub enum ApplyError {
    RuleNotDefined,
    RuntimeError(Value, LeftOrRight, Operator2),
}

type ApplyResult = Result<DerivationNode, ApplyError>;

pub fn eval(expr: &Expr) -> ApplyResult {
    match expr {
        Expr::Int(i) => Ok(DerivationNode::Expr(ExprNode::new(
            expr.clone(),
            Value::Int(*i),
            Rule::EInt,
            vec![],
        ))),
        Expr::Bool(b) => Ok(DerivationNode::Expr(ExprNode::new(
            expr.clone(),
            Value::Bool(*b),
            Rule::EBool,
            vec![],
        ))),
        Expr::Operand2(e1, op, e2) => {
            let e1 = eval(e1)?;
            let e2 = eval(e2)?;
            let left_value = e1.eval_to().unwrap();
            let right_value = e2.eval_to().unwrap();

            let eval_to = match op.eval(left_value, right_value) {
                Ok(v) => v,
                Err(ApplyError::RuntimeError(v, left_or_right, op)) => {
                    return Ok(DerivationNode::Error {
                        expr: expr.clone(),
                        rule: match (v, &left_or_right, op) {
                            (Value::Bool(_), LeftOrRight::Left, Operator2::Add) => Rule::EPlusBoolL,
                            (Value::Bool(_), LeftOrRight::Right, Operator2::Add) => {
                                Rule::EPlusBoolR
                            }
                            (Value::Bool(_), LeftOrRight::Left, Operator2::Sub) => {
                                Rule::EMinusBoolL
                            }
                            (Value::Bool(_), LeftOrRight::Right, Operator2::Sub) => {
                                Rule::EMinusBoolR
                            }
                            (Value::Bool(_), LeftOrRight::Left, Operator2::Mul) => {
                                Rule::ETimesBoolL
                            }
                            (Value::Bool(_), LeftOrRight::Right, Operator2::Mul) => {
                                Rule::ETimesBoolR
                            }
                            (Value::Bool(_), LeftOrRight::Left, Operator2::Lt) => Rule::ELtBoolL,
                            (Value::Bool(_), LeftOrRight::Right, Operator2::Lt) => Rule::ELtBoolR,
                            (Value::Error, LeftOrRight::Left, Operator2::Add) => Rule::EPlusErrorL,
                            (Value::Error, LeftOrRight::Right, Operator2::Add) => Rule::EPlusErrorR,
                            _ => Err(ApplyError::RuleNotDefined)?,
                        },
                        children: vec![match left_or_right {
                            LeftOrRight::Left => Box::new(e1.clone()),
                            LeftOrRight::Right => Box::new(e2.clone()),
                        }],
                    })
                }
                _ => unreachable!(),
            };
            Ok(DerivationNode::Expr(ExprNode::new(
                expr.clone(),
                eval_to.clone(),
                op.to_expr_rule().unwrap(),
                vec![
                    Box::new(e1.clone()),
                    Box::new(e2.clone()),
                    Box::new(DerivationNode::Binary {
                        rule: op.to_binary_rule().unwrap(),
                        left: left_value.clone(),
                        operand: op.to_alpha_string(),
                        right: right_value.clone(),
                        answer: eval_to.clone(),
                    }),
                ],
            )))
        }
        Expr::If(e1, e2, e3) => {
            let e1 = eval(e1)?;
            let Some(cond) = e1.eval_to().unwrap().bool() else {
                return Ok(DerivationNode::Error {
                    expr: expr.clone(),
                    rule: Rule::EIfInt,
                    children: vec![Box::new(e1.clone())],
                });
            };
            if cond {
                let a = eval(e2)?;
                match a.eval_to().unwrap() {
                    Value::Error => Ok(DerivationNode::Error {
                        expr: expr.clone(),
                        rule: Rule::EIfTError,
                        children: vec![Box::new(e1.clone()), Box::new(a)],
                    }),
                    v => Ok(DerivationNode::Expr(ExprNode::new(
                        expr.clone(),
                        v.clone(),
                        Rule::EIfT,
                        vec![Box::new(e1.clone()), Box::new(a)],
                    ))),
                }
            } else {
                let a = eval(e3)?;
                Ok(DerivationNode::Expr(ExprNode::new(
                    expr.clone(),
                    a.eval_to().unwrap().clone(),
                    Rule::EIfF,
                    vec![Box::new(e1.clone()), Box::new(a)],
                )))
            }
        }
        _ => Err(ApplyError::RuleNotDefined),
    }
}

impl Display for DerivationNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DerivationNode::Expr(expr_node) => {
                write!(
                    f,
                    "{} evalto {} by {} {{",
                    &expr_node.expr, &expr_node.eval_to, &expr_node.rule
                )?;
                for (i, child) in expr_node.children.iter().enumerate() {
                    if i > 0 {
                        write!(f, ";")?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, "}}")?;
            }
            DerivationNode::Error {
                expr,
                rule,
                children,
            } => {
                write!(f, "{} evalto error by {} {{", expr, rule)?;
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, ";")?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, "}}")?;
            }
            DerivationNode::Binary {
                left,
                operand,
                right,
                answer,
                rule,
            } => {
                write!(
                    f,
                    "{} {} {} is {} by {} {{}}",
                    left, operand, right, answer, rule
                )?;
            }
        }

        Ok(())
    }
}

pub enum Error {}

struct ParserResult {}

#[derive(Debug, PartialEq)]
pub enum Token {
    LeftParen,
    RightParen,
    Plus,
    Minus,
    Times,
    LessThan,
    Equals,
    If,
    Then,
    Else,
    Let,
    In,
    Int(i32),
    Bool(bool),
    Ident(String),
}

pub fn tokenize(input: &str) -> anyhow::Result<Vec<Token>> {
    let mut tokens = vec![];
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            ' ' => continue,
            '(' => tokens.push(Token::LeftParen),
            ')' => tokens.push(Token::RightParen),
            '+' => tokens.push(Token::Plus),
            '-' => {
                let mut num = "".to_string();
                while let Some('0'..='9') = chars.peek() {
                    num.push(chars.next().unwrap());
                }
                if num.is_empty() {
                    tokens.push(Token::Minus)
                } else {
                    tokens.push(Token::Int(-num.parse::<i32>()?));
                }
            }
            '*' => tokens.push(Token::Times),
            '<' => tokens.push(Token::LessThan),
            '=' => tokens.push(Token::Equals),
            '0'..='9' => {
                let mut num = c.to_string();
                while let Some('0'..='9') = chars.peek() {
                    num.push(chars.next().unwrap());
                }
                tokens.push(Token::Int(num.parse::<i32>()?));
            }
            'a'..='z' | 'A'..='Z' => {
                let mut s = c.to_string();
                s.extend(chars.take_while_ref(|c| c.is_alphanumeric() || *c == '_'));
                match s.as_str() {
                    "if" => tokens.push(Token::If),
                    "then" => tokens.push(Token::Then),
                    "else" => tokens.push(Token::Else),
                    "let" => tokens.push(Token::Let),
                    "in" => tokens.push(Token::In),
                    "true" => tokens.push(Token::Bool(true)),
                    "false" => tokens.push(Token::Bool(false)),
                    _ => tokens.push(Token::Ident(s)),
                }
            }
            _ => bail!("Invalid token {:?}", chars.collect::<String>()),
        }
    }

    Ok(tokens)
}

#[derive(Debug)]
pub enum ParseResult<'a, T> {
    Ok(T, &'a [Token]),
    Err(&'a [Token]),
}

#[derive(Debug)]
pub enum ParseVecResult<'a> {
    Ok(Vec<Expr>, &'a [Token]),
    Err(&'a [Token]),
}

impl<'a, T> ParseResult<'a, T> {
    pub fn unwrap(self) -> T {
        match self {
            ParseResult::Ok(expr, _) => expr,
            ParseResult::Err(tokens) => panic!("Parse error: {:?}", tokens),
        }
    }
}

type ParserFunc<'a, T> = Box<dyn Fn(&'a [Token]) -> ParseResult<'a, T> + 'a>;

pub fn one_of<'a, T: 'static>(parsers: Vec<ParserFunc<'a, T>>) -> ParserFunc<'a, T> {
    Box::new(move |tokens| {
        for parser in &parsers {
            match parser(tokens) {
                ParseResult::Ok(expr, rest) => return ParseResult::Ok(expr, rest),
                _ => continue,
            }
        }
        ParseResult::Err(tokens)
    })
}

pub fn parse_int(tokens: &[Token]) -> ParseResult<Expr> {
    match tokens {
        [Token::Int(i), rest @ ..] => ParseResult::Ok(Expr::Int(*i), rest),
        _ => ParseResult::Err(tokens),
    }
}

pub fn parse_bool(tokens: &[Token]) -> ParseResult<Expr> {
    match tokens {
        [Token::Bool(b), rest @ ..] => ParseResult::Ok(Expr::Bool(*b), rest),
        _ => ParseResult::Err(tokens),
    }
}

pub fn parse_expr(tokens: &[Token]) -> ParseResult<Expr> {
    one_of(vec![
        Box::new(parse_less_than),
        Box::new(parse_if),
        Box::new(parse_let_in),
    ])(tokens)
}

pub fn parse_unary_expr(tokens: &[Token]) -> ParseResult<Expr> {
    one_of(vec![
        Box::new(parse_paren_expr),
        Box::new(parse_int),
        Box::new(parse_bool),
        map(Box::new(parse_ident), |id| Expr::Variable(id)),
    ])(tokens)
}

pub fn parse_times(tokens: &[Token]) -> ParseResult<Expr> {
    map(
        sequence(vec![
            Seq::Expr(Box::new(parse_unary_expr)),
            Seq::VecExpr(many0(map(
                sequence(vec![
                    Seq::Op(one_of(vec![tag_op(Operator2::Mul)])),
                    Seq::Expr(Box::new(parse_unary_expr)),
                ]),
                |parsed| (parsed[0].op().clone(), parsed[1].expr().clone()),
            ))),
        ]),
        |exprs| {
            let mut expr = exprs[0].expr().clone();
            for (op, expr2) in exprs[1].exprs() {
                expr = Expr::Operand2(Box::new(expr), *op, Box::new(expr2.clone()));
            }
            expr
        },
    )(tokens)
}

pub fn parse_less_than(tokens: &[Token]) -> ParseResult<Expr> {
    one_of(vec![
        map(
            sequence(vec![
                Seq::Expr(Box::new(parse_add_sub)),
                Seq::Op(tag_op(Operator2::Lt)),
                Seq::Expr(Box::new(parse_add_sub)),
            ]),
            |exprs| {
                Expr::Operand2(
                    Box::new(exprs[0].expr().clone()),
                    exprs[1].op().clone(),
                    Box::new(exprs[2].expr().clone()),
                )
            },
        ),
        Box::new(parse_add_sub),
    ])(tokens)
}

pub fn many0<'a, T: 'a>(parser: ParserFunc<'a, T>) -> ParserFunc<'a, Vec<T>> {
    Box::new(move |tokens| {
        let mut results = vec![];
        let mut rest = tokens;
        loop {
            match parser(rest) {
                ParseResult::Ok(expr, rest2) => {
                    results.push(expr);
                    rest = rest2;
                }
                _ => break,
            }
        }
        ParseResult::Ok(results, rest)
    })
}

pub fn parse_add_sub(tokens: &[Token]) -> ParseResult<Expr> {
    map(
        sequence(vec![
            Seq::Expr(Box::new(parse_times)),
            Seq::VecExpr(many0(map(
                sequence(vec![
                    Seq::Op(one_of(vec![tag_op(Operator2::Add), tag_op(Operator2::Sub)])),
                    Seq::Expr(Box::new(parse_times)),
                ]),
                |parsed| (parsed[0].op().clone(), parsed[1].expr().clone()),
            ))),
        ]),
        |exprs| {
            let mut expr = exprs[0].expr().clone();
            for (op, expr2) in exprs[1].exprs() {
                expr = Expr::Operand2(Box::new(expr), *op, Box::new(expr2.clone()));
            }
            expr
        },
    )(tokens)
}

pub fn parse_paren_expr(tokens: &[Token]) -> ParseResult<Expr> {
    map(
        sequence(vec![
            Seq::Skip(tag(Token::LeftParen)),
            Seq::Expr(Box::new(parse_expr)),
            Seq::Skip(tag(Token::RightParen)),
        ]),
        |exprs| exprs[0].expr().clone(),
    )(tokens)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ident(String);

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub enum Seq<'a> {
    Skip(ParserFunc<'a, ()>),
    Expr(ParserFunc<'a, Expr>),
    Ident(ParserFunc<'a, Ident>),
    VecExpr(ParserFunc<'a, Vec<(Operator2, Expr)>>),
    Op(ParserFunc<'a, Operator2>),
}

#[derive(Debug, Clone)]
pub enum SeqValue {
    Expr(Expr),
    Op(Operator2),
    VecExpr(Vec<(Operator2, Expr)>),
    Ident(Ident),
}

impl SeqValue {
    pub fn expr(&self) -> &Expr {
        match self {
            SeqValue::Expr(expr) => expr,
            _ => panic!("Not an expr {:?}", self),
        }
    }

    pub fn exprs(&self) -> &Vec<(Operator2, Expr)> {
        match self {
            SeqValue::VecExpr(exprs) => exprs,
            _ => panic!("Not an expr"),
        }
    }

    pub fn op(&self) -> &Operator2 {
        match self {
            SeqValue::Op(op) => op,
            _ => panic!("Not an op"),
        }
    }

    pub fn ident(&self) -> &Ident {
        match self {
            SeqValue::Ident(ident) => ident,
            _ => panic!("Not an ident"),
        }
    }
}

pub fn map<'a, I: 'a, O: 'a>(parser: ParserFunc<'a, I>, f: fn(I) -> O) -> ParserFunc<'a, O> {
    Box::new(move |tokens| match parser(tokens) {
        ParseResult::Ok(expr, rest) => ParseResult::Ok(f(expr), rest),
        ParseResult::Err(rest) => ParseResult::Err(rest),
    })
}

pub fn sequence<'a>(parsers: Vec<Seq<'a>>) -> ParserFunc<Vec<SeqValue>> {
    Box::new(move |tokens| {
        let mut results = vec![];
        let mut rest = tokens;
        for parser in &parsers {
            match parser {
                Seq::Skip(parser) => match parser(rest) {
                    ParseResult::Ok(_, rest2) => {
                        rest = rest2;
                    }
                    _ => return ParseResult::Err(tokens),
                },
                Seq::Expr(parser) => match parser(rest) {
                    ParseResult::Ok(expr, rest2) => {
                        rest = rest2;
                        results.push(SeqValue::Expr(expr));
                    }
                    _ => return ParseResult::Err(tokens),
                },
                Seq::Op(parser) => match parser(rest) {
                    ParseResult::Ok(op, rest2) => {
                        rest = rest2;
                        results.push(SeqValue::Op(op));
                    }
                    _ => return ParseResult::Err(tokens),
                },
                Seq::VecExpr(parser) => match parser(rest) {
                    ParseResult::Ok(exprs, rest2) => {
                        rest = rest2;
                        results.push(SeqValue::VecExpr(exprs));
                    }
                    _ => return ParseResult::Err(tokens),
                },
                Seq::Ident(parser) => match parser(rest) {
                    ParseResult::Ok(ident, rest2) => {
                        rest = rest2;
                        results.push(SeqValue::Ident(ident));
                    }
                    _ => return ParseResult::Err(tokens),
                },
            }
        }
        ParseResult::Ok(results, rest)
    })
}

pub fn tag<'a>(tag: Token) -> ParserFunc<'a, ()> {
    Box::new(move |tokens| match tokens {
        [t, rest @ ..] if *t == tag => ParseResult::Ok((), rest),
        _ => ParseResult::Err(tokens),
    })
}

pub fn tag_op<'a>(op: Operator2) -> ParserFunc<'a, Operator2> {
    let token = match op {
        Operator2::Add => Token::Plus,
        Operator2::Sub => Token::Minus,
        Operator2::Mul => Token::Times,
        Operator2::Lt => Token::LessThan,
    };
    Box::new(move |tokens| match tokens {
        [tok, rest @ ..] if tok == &token => ParseResult::Ok(op, rest),
        _ => ParseResult::Err(tokens),
    })
}

pub fn parse_if(token: &[Token]) -> ParseResult<Expr> {
    match sequence(vec![
        Seq::Skip(tag(Token::If)),
        Seq::Expr(Box::new(parse_expr)),
        Seq::Skip(tag(Token::Then)),
        Seq::Expr(Box::new(parse_expr)),
        Seq::Skip(tag(Token::Else)),
        Seq::Expr(Box::new(parse_expr)),
    ])(token)
    {
        ParseResult::Ok(exprs, rest) => ParseResult::Ok(
            Expr::If(
                Box::new(exprs[0].expr().clone()),
                Box::new(exprs[1].expr().clone()),
                Box::new(exprs[2].expr().clone()),
            ),
            rest,
        ),
        ParseResult::Err(rest) => ParseResult::Err(rest),
    }
}

pub fn parse_ident(token: &[Token]) -> ParseResult<Ident> {
    match token {
        [Token::Ident(s), rest @ ..] => ParseResult::Ok(Ident(s.clone()), rest),
        _ => ParseResult::Err(token),
    }
}

pub fn parse_let_in(token: &[Token]) -> ParseResult<Expr> {
    match sequence(vec![
        Seq::Skip(tag(Token::Let)),
        Seq::Ident(Box::new(parse_ident)),
        Seq::Skip(tag(Token::Equals)),
        Seq::Expr(Box::new(parse_expr)),
        Seq::Skip(tag(Token::In)),
        Seq::Expr(Box::new(parse_expr)),
    ])(token)
    {
        ParseResult::Ok(exprs, rest) => ParseResult::Ok(
            Expr::Let(
                exprs[0].ident().clone(),
                Box::new(exprs[1].expr().clone()),
                Box::new(exprs[2].expr().clone()),
            ),
            rest,
        ),
        ParseResult::Err(rest) => ParseResult::Err(rest),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_add() {
        let expr_str = "1 + 2 + 3";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Operand2(
                Box::new(Expr::Operand2(
                    Box::new(Expr::Int(1)),
                    Operator2::Add,
                    Box::new(Expr::Int(2)),
                )),
                Operator2::Add,
                Box::new(Expr::Int(3)),
            )
        );
    }

    #[test]
    fn parse_sub() {
        let expr_str = "1 - 2 - 3";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Operand2(
                Box::new(Expr::Operand2(
                    Box::new(Expr::Int(1)),
                    Operator2::Sub,
                    Box::new(Expr::Int(2)),
                )),
                Operator2::Sub,
                Box::new(Expr::Int(3)),
            )
        );
    }

    #[test]
    fn parse_times() {
        let expr_str = "1 * 2 * 3";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Operand2(
                Box::new(Expr::Operand2(
                    Box::new(Expr::Int(1)),
                    Operator2::Mul,
                    Box::new(Expr::Int(2)),
                )),
                Operator2::Mul,
                Box::new(Expr::Int(3)),
            )
        );
    }

    #[test]
    fn parse_times2() {
        let expr_str = "1 + 2 * 3";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Operand2(
                Box::new(Expr::Int(1)),
                Operator2::Add,
                Box::new(Expr::Operand2(
                    Box::new(Expr::Int(2)),
                    Operator2::Mul,
                    Box::new(Expr::Int(3)),
                )),
            )
        );
    }

    #[test]
    fn parse_times3() {
        let expr_str = "(1 + 2) * 3";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Operand2(
                Box::new(Expr::Operand2(
                    Box::new(Expr::Int(1)),
                    Operator2::Add,
                    Box::new(Expr::Int(2)),
                )),
                Operator2::Mul,
                Box::new(Expr::Int(3)),
            )
        );
    }

    #[test]
    fn parse_let() {
        let expr_str = "let x = 1 in x + 2";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Let(
                Ident("x".to_string()),
                Box::new(Expr::Int(1)),
                Box::new(Expr::Operand2(
                    Box::new(Expr::Variable(Ident("x".to_string()))),
                    Operator2::Add,
                    Box::new(Expr::Int(2)),
                )),
            )
        );
    }
}
