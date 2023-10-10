use anyhow::bail;
use itertools::Itertools;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::ops::Sub;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i32),
    Bool(bool),
    Fun(Box<Environment>, Ident, Box<Expr>),
    Error,
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Fun(environ, ident, expr) => {
                write!(f, "({}) [fun {} -> {}]", environ, ident, expr)
            }
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

    pub fn fun(&self) -> Option<(Box<Environment>, Ident, Box<Expr>)> {
        match self {
            Value::Fun(environ, ident, expr) => {
                Some((environ.clone(), ident.clone(), expr.clone()))
            }
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
    Fun(Ident, Box<Expr>),
    Apply(Box<Expr>, Box<Expr>),
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
            Expr::Fun(s, e) => write!(f, "(fun {} -> {})", s, e),
            Expr::Apply(e1, e2) => write!(f, "({} {})", e1, e2),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Rule {
    EInt,
    EBool,
    EVar1,
    EVar2,
    EPlus,
    EMinus,
    ETimes,
    ELt,
    EIfT,
    EIfF,
    ELet,
    EFun,
    EApp,
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
            Rule::ELet => write!(f, "E-Let"),
            Rule::EFun => write!(f, "E-Fun"),
            Rule::EApp => write!(f, "E-App"),
            Rule::EIfT => write!(f, "E-IfT"),
            Rule::EIfF => write!(f, "E-IfF"),
            Rule::ELt => write!(f, "E-Lt"),
            Rule::BLt => write!(f, "B-Lt"),
            Rule::EVar1 => write!(f, "E-Var1"),
            Rule::EVar2 => write!(f, "E-Var2"),
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
    environment: Environment,
}

impl ExprNode {
    pub fn new(
        expr: Expr,
        eval_to: Value,
        rule: Rule,
        children: Vec<Box<DerivationNode>>,
        environment: Environment,
    ) -> Self {
        Self {
            expr,
            eval_to,
            rule,
            children,
            environment,
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
    VariableNotFound(Ident),
}

type ApplyResult = Result<DerivationNode, ApplyError>;

#[derive(Debug, Clone)]
pub enum Environment {
    Extend(Ident, Value, Box<Environment>),
    Empty,
}

impl FromIterator<(Ident, Value)> for Environment {
    fn from_iter<T: IntoIterator<Item = (Ident, Value)>>(iter: T) -> Self {
        let mut env = Environment::Empty;
        for (id, v) in iter {
            env = Environment::Extend(id, v, Box::new(env));
        }
        env
    }
}

impl Display for Environment {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Environment::Extend(id, v, env) if matches!(env.as_ref(), Environment::Empty) => {
                write!(f, "{} = {}", id, v)
            }
            Environment::Extend(id, v, env) => {
                write!(f, "{}, {} = {}", env, id, v)
            }
            Environment::Empty => write!(f, ""),
        }
    }
}

pub fn eval(environment: &Environment, expr: &Expr) -> ApplyResult {
    match expr {
        Expr::Int(i) => Ok(DerivationNode::Expr(ExprNode::new(
            expr.clone(),
            Value::Int(*i),
            Rule::EInt,
            vec![],
            environment.clone(),
        ))),
        Expr::Bool(b) => Ok(DerivationNode::Expr(ExprNode::new(
            expr.clone(),
            Value::Bool(*b),
            Rule::EBool,
            vec![],
            environment.clone(),
        ))),
        Expr::Variable(ident) => match &environment {
            Environment::Extend(id, v, env) => {
                if id == ident {
                    Ok(DerivationNode::Expr(ExprNode::new(
                        expr.clone(),
                        v.clone(),
                        Rule::EVar1,
                        vec![],
                        environment.clone(),
                    )))
                } else {
                    let e = eval(env, expr)?;
                    Ok(DerivationNode::Expr(ExprNode::new(
                        expr.clone(),
                        e.eval_to().unwrap().clone(),
                        Rule::EVar2,
                        vec![Box::new(e)],
                        environment.clone(),
                    )))
                }
            }
            _ => Err(ApplyError::VariableNotFound(ident.clone())),
        },
        Expr::Operand2(e1, op, e2) => {
            let e1 = eval(environment, e1)?;
            let e2 = eval(environment, e2)?;
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
                environment.clone(),
            )))
        }
        Expr::If(e1, e2, e3) => {
            let e1 = eval(environment, e1)?;
            let Some(cond) = e1.eval_to().unwrap().bool() else {
                return Ok(DerivationNode::Error {
                    expr: expr.clone(),
                    rule: Rule::EIfInt,
                    children: vec![Box::new(e1.clone())],
                });
            };
            if cond {
                let a = eval(environment, e2)?;
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
                        environment.clone(),
                    ))),
                }
            } else {
                let a = eval(environment, e3)?;
                Ok(DerivationNode::Expr(ExprNode::new(
                    expr.clone(),
                    a.eval_to().unwrap().clone(),
                    Rule::EIfF,
                    vec![Box::new(e1.clone()), Box::new(a)],
                    environment.clone(),
                )))
            }
        }
        Expr::Let(ident, e1, e2) => {
            let e1 = eval(environment, e1)?;
            let e2 = eval(
                &Environment::Extend(
                    ident.clone(),
                    e1.eval_to().unwrap().clone(),
                    Box::new(environment.clone()),
                ),
                e2,
            )?;
            Ok(DerivationNode::Expr(ExprNode::new(
                expr.clone(),
                e2.eval_to().unwrap().clone(),
                Rule::ELet,
                vec![Box::new(e1), Box::new(e2)],
                environment.clone(),
            )))
        }
        Expr::Fun(ident, e) => Ok(DerivationNode::Expr(ExprNode::new(
            expr.clone(),
            Value::Fun(Box::new(environment.clone()), ident.clone(), e.clone()),
            Rule::EFun,
            vec![],
            environment.clone(),
        ))),
        Expr::Apply(func, arg) => {
            let e1 = eval(environment, func)?;
            let e2 = eval(environment, arg)?;
            let (closed_env, ident, func_body) = e1.eval_to().unwrap().fun().unwrap();
            let e3 = eval(
                &Environment::Extend(ident.clone(), e2.eval_to().unwrap().clone(), closed_env),
                &func_body,
            )?;
            Ok(DerivationNode::Expr(ExprNode::new(
                expr.clone(),
                e3.eval_to().unwrap().clone(),
                Rule::EApp,
                vec![Box::new(e1), Box::new(e2), Box::new(e3)],
                environment.clone(),
            )))
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
                    "{} |- {} evalto {} by {} {{",
                    &expr_node.environment, &expr_node.expr, &expr_node.eval_to, &expr_node.rule
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
    Fun,
    Arrow,
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
                    if let Some('>') = chars.peek() {
                        chars.next();
                        tokens.push(Token::Arrow);
                    } else {
                        tokens.push(Token::Minus)
                    }
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
                    "fun" => tokens.push(Token::Fun),
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
            ParseResult::Ok(expr, remain) => {
                assert!(remain.is_empty(), "Remain is not empty: {:?}", remain);
                expr
            }
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
        Box::new(parse_fn_expr),
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

pub fn parse_fn_expr(tokens: &[Token]) -> ParseResult<Expr> {
    map(
        sequence(vec![
            Seq::Skip(tag(Token::Fun)),
            Seq::Ident(Box::new(parse_ident)),
            Seq::Skip(tag(Token::Arrow)),
            Seq::Expr(Box::new(parse_expr)),
        ]),
        |exps| Expr::Fun(exps[0].ident().clone(), Box::new(exps[1].expr().clone())),
    )(tokens)
}

pub fn parse_apply(tokens: &[Token]) -> ParseResult<Expr> {
    map(
        sequence(vec![
            Seq::Expr(Box::new(parse_unary_expr)),
            Seq::VecExpr(many0(Box::new(parse_unary_expr))),
        ]),
        |exprs| {
            let mut expr = exprs[0].expr().clone();
            for arg in exprs[1].exprs() {
                expr = Expr::Apply(Box::new(expr), Box::new(arg.clone()));
            }
            expr
        },
    )(tokens)
}

pub fn parse_times(tokens: &[Token]) -> ParseResult<Expr> {
    map(
        sequence(vec![
            Seq::Expr(Box::new(parse_apply)),
            Seq::VecOpExpr(many0(map(
                sequence(vec![
                    Seq::Op(one_of(vec![tag_op(Operator2::Mul)])),
                    Seq::Expr(Box::new(parse_apply)),
                ]),
                |parsed| (parsed[0].op().clone(), parsed[1].expr().clone()),
            ))),
        ]),
        |exprs| {
            let mut expr = exprs[0].expr().clone();
            for (op, expr2) in exprs[1].op_exprs() {
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
            Seq::VecOpExpr(many0(map(
                sequence(vec![
                    Seq::Op(one_of(vec![tag_op(Operator2::Add), tag_op(Operator2::Sub)])),
                    Seq::Expr(Box::new(parse_times)),
                ]),
                |parsed| (parsed[0].op().clone(), parsed[1].expr().clone()),
            ))),
        ]),
        |exprs| {
            let mut expr = exprs[0].expr().clone();
            for (op, expr2) in exprs[1].op_exprs() {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ident(String);

impl Ident {
    pub fn new(field0: &str) -> Self {
        Self(field0.to_string())
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub enum Seq<'a> {
    Skip(ParserFunc<'a, ()>),
    Expr(ParserFunc<'a, Expr>),
    Ident(ParserFunc<'a, Ident>),
    VecOpExpr(ParserFunc<'a, Vec<(Operator2, Expr)>>),
    VecExpr(ParserFunc<'a, Vec<Expr>>),
    OptionExpr(ParserFunc<'a, Option<Expr>>),
    Op(ParserFunc<'a, Operator2>),
}

#[derive(Debug, Clone)]
pub enum SeqValue {
    Expr(Expr),
    Op(Operator2),
    VecOpExpr(Vec<(Operator2, Expr)>),
    VecExpr(Vec<Expr>),
    Ident(Ident),
    OptionExpr(Option<Expr>),
}

impl SeqValue {
    pub fn expr(&self) -> &Expr {
        match self {
            SeqValue::Expr(expr) => expr,
            _ => panic!("Not an expr {:?}", self),
        }
    }

    pub fn op_exprs(&self) -> &Vec<(Operator2, Expr)> {
        match self {
            SeqValue::VecOpExpr(exprs) => exprs,
            _ => panic!("Not an expr"),
        }
    }

    pub fn exprs(&self) -> &Vec<Expr> {
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

    pub fn option_expr(&self) -> &Option<Expr> {
        match self {
            SeqValue::OptionExpr(expr) => expr,
            _ => panic!("Not an option expr"),
        }
    }
}

pub fn map<'a, I: 'a, O: 'a>(parser: ParserFunc<'a, I>, f: fn(I) -> O) -> ParserFunc<'a, O> {
    Box::new(move |tokens| match parser(tokens) {
        ParseResult::Ok(expr, rest) => ParseResult::Ok(f(expr), rest),
        ParseResult::Err(rest) => ParseResult::Err(rest),
    })
}

pub fn opt<'a>(parser: ParserFunc<'a, Expr>) -> ParserFunc<'a, Option<Expr>> {
    Box::new(move |tokens| match parser(tokens) {
        ParseResult::Ok(expr, rest) => ParseResult::Ok(Some(expr), rest),
        ParseResult::Err(_) => ParseResult::Ok(None, tokens),
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
                Seq::VecOpExpr(parser) => match parser(rest) {
                    ParseResult::Ok(exprs, rest2) => {
                        rest = rest2;
                        results.push(SeqValue::VecOpExpr(exprs));
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
                Seq::OptionExpr(parser) => match parser(rest) {
                    ParseResult::Ok(expr, rest2) => {
                        rest = rest2;
                        results.push(SeqValue::OptionExpr(expr));
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

// pub fn free_variables(expr: &Expr) -> BTreeSet<Ident> {
//     match expr {
//         Expr::Int(_) => BTreeSet::new(),
//         Expr::Bool(_) => BTreeSet::new(),
//         Expr::Variable(ident) => {
//             let mut set = BTreeSet::new();
//             set.insert(ident.clone());
//             set
//         }
//         Expr::Operand2(e1, _, e2) => {
//             let mut set = free_variables(e1);
//             set.extend(free_variables(e2));
//             set
//         }
//         Expr::If(e1, e2, e3) => {
//             let mut set = free_variables(e1);
//             set.extend(free_variables(e2));
//             set.extend(free_variables(e3));
//             set
//         }
//         Expr::Let(ident, e1, e2) => {
//             let mut set = free_variables(e1);
//             let mut set2 = free_variables(e2);
//             set2.remove(ident);
//             set.extend(set2);
//             set
//         }
//     }
// }

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

    #[test]
    fn parse_fn() {
        let expr_str = "fun x -> x + 1";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Fun(
                Ident("x".to_string()),
                Box::new(Expr::Operand2(
                    Box::new(Expr::Variable(Ident("x".to_string()))),
                    Operator2::Add,
                    Box::new(Expr::Int(1)),
                )),
            )
        );
    }

    #[test]
    fn parse_apply() {
        let expr_str = "let f = fun x -> x + 1 in f 10";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Let(
                Ident("f".to_string()),
                Box::new(Expr::Fun(
                    Ident("x".to_string()),
                    Box::new(Expr::Operand2(
                        Box::new(Expr::Variable(Ident("x".to_string()))),
                        Operator2::Add,
                        Box::new(Expr::Int(1)),
                    )),
                )),
                Box::new(Expr::Apply(
                    Box::new(Expr::Variable(Ident("f".to_string()))),
                    Box::new(Expr::Int(10)),
                )),
            )
        );
    }

    #[test]
    fn parse_apply2() {
        let expr_str = "max 1 2";

        let tokens = tokenize(expr_str).unwrap();
        let expr = parse_expr(&tokens).unwrap();

        assert_eq!(
            expr,
            Expr::Apply(
                Box::new(Expr::Apply(
                    Box::new(Expr::Variable(Ident("max".to_string()))),
                    Box::new(Expr::Int(1)),
                )),
                Box::new(Expr::Int(2)),
            )
        );
    }
}
