use std::fmt::{Display, Write};

use super::Matrix;

fn get_number_width(num: f32) -> u32 {
    let mut number = num.floor() as i64;
    if number == 0 {
        2
    } else {
        let mut dimension = 1; //if number < 0 { 1 } else { 0 };
        number = number.abs();
        while number > 0 {
            dimension += 1;
            number /= 10;
        }

        dimension
    }
}

fn get_column_width(matrix: &Matrix, x: usize) -> u32 {
    let mut width = 0;
    for y in 0..matrix.size[0] {
        let number = matrix[[y, x]];
        let number_width = get_number_width(number);
        width = width.max(number_width);
    }

    width
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision: usize = match f.precision() {
            Some(p) => p,
            None => 2,
        };

        let mut max_width = 0;
        for x in 0..self.size[1] {
            max_width = max_width.max(get_column_width(self, x));
        }

        let precision_character_count = if precision > 0 {
            precision as u32 + 1
        } else {
            0
        };

        let line_character_count =
            self.size[1] as u32 * (max_width + precision_character_count + 2);

        let mut result = String::with_capacity(
            ((line_character_count + 2) * (self.size[0] as u32 + 2)) as usize,
        );

        result += " ";
        for _ in 0..line_character_count {
            result += "-";
        }
        result += " \n";

        for y in 0..self.size[0] {
            result += "| ";

            for x in 0..self.size[1] {
                let number = self[[y, x]];
                let width = get_number_width(number);
                write!(
                    result,
                    "{}{:.*}",
                    if number < 0.0 { "" } else { " " },
                    precision,
                    number
                )?;

                for _ in 0..(max_width - width) {
                    result += " ";
                }

                if x == (self.size[1] - 1) {
                    result += " ";
                } else {
                    result += "  ";
                }
            }
            result += "|\n";
        }
        result += " ";
        for _ in 0..line_character_count {
            result += "-";
        }
        result += " \n";

        write!(f, "{}", result)
    }
}
