use min_caml::{eval, parse_expr, tokenize};

fn main() {
    // let expr = Expr::Operand2(
    //     Box::new(Expr::Int(3)),
    //     Operator2::Add,
    //     Box::new(Expr::Int(5)),
    // );
    // let dt = eval(&expr).unwrap();
    // println!("{}", dt);
    //
    // let expr = Expr::Operand2(
    //     Box::new(Expr::Operand2(
    //         Box::new(Expr::Int(8)),
    //         Operator2::Sub,
    //         Box::new(Expr::Int(2)),
    //     )),
    //     Operator2::Sub,
    //     Box::new(Expr::Int(3)),
    // );
    // let dt = eval(&expr).unwrap();
    // println!("{}", dt);
    //
    // let tokens = tokenize("1 + 2").unwrap();
    // let ast = parse_expr(&tokens);
    // dbg!(ast);
    //
    // let tokens = tokenize("1 * 2 + 3").unwrap();
    // let ast = parse_expr(&tokens);
    // dbg!(ast);
    //
    // let tokens = tokenize("(4 + 5) * (1 - 10)").unwrap();
    // let expr = parse_expr(&tokens).unwrap();
    //
    // let dt = eval(&expr).unwrap();
    // println!("{}", dt);
    //
    // let tokens = tokenize("if 4 < 5 then 2 + 3 else 8 * 8").unwrap();
    // let expr = parse_expr(&tokens).unwrap();
    // let dt = eval(&expr).unwrap();
    // println!("{}", dt);
    //
    // let tokens = tokenize("3 + if -23 < -2 * 8 then 8 else 2 + 4").unwrap();
    // let expr = parse_expr(&tokens).unwrap();
    // let dt = eval(&expr).unwrap();
    // println!("{}", dt);
    //
    let tokens = tokenize("3 + (if -23 < -2 * 8 then 8 else 2) + 4").unwrap();
    let expr = parse_expr(&tokens).unwrap();
    let dt = eval(&expr).unwrap();
    println!("{}", dt);

    let tokens = tokenize("1 + true + 2").unwrap();
    let expr = parse_expr(&tokens).unwrap();

    let dt = eval(&expr).unwrap();
    println!("{}", dt);

    let tokens = tokenize("if 2 + 3 then 1 else 3").unwrap();
    let expr = parse_expr(&tokens).unwrap();

    let dt = eval(&expr).unwrap();
    println!("{}", dt);

    let tokens = tokenize("if 3 < 4 then 1 < true else 3 - false").unwrap();
    let expr = parse_expr(&tokens).unwrap();

    let dt = eval(&expr).unwrap();
    println!("{}", dt);
}
