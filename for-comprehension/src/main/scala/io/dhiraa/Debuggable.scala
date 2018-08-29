package io.dhiraa

case class DebuggableWrapper[A](value: A, msg: String) {

  def map[B](f: A => B): DebuggableWrapper[B] = {
    println(">>>>>>>  "+ this.toString + " map "+ value + " => ")
    val newValue = f(value)
    println("<<<<<<<  " + this.toString + " map \n")
    DebuggableWrapper(newValue, msg)
  }

  def flatMap[B](f: A => DebuggableWrapper[B]): DebuggableWrapper[B] = {
    println(">>>>>>>  "+ this.toString + " flatmap " + this.toString + " map "+ value + " => ")
    val newValue = f(value)
    val res = DebuggableWrapper(newValue.value, msg + " " + newValue.msg)
    println("<<<<<<< "+ this.toString + " flatmap " +  res + "\n")
    res
  }
}

case class State(value: Int) {

  def flatMap(f: Int => State): State = {
    println(">>>>>>>  "+ this.toString + " flatmap " + "\n")
    val newState = f(value)
    println("<<<<<<<  " + this.toString + " flatmap \n")
    State(newState.value)
  }

  def map(f: Int => Int) = {
    println(">>>>>>>  "+ this.toString + " map " + "\n")
    val res = State(f(value))
    println("<<<<<<<  " + this.toString + " map \n")
    res
  }
}

object Debuggable {

  //Lets see how we can bind/compbine pure functions in FP style using Higher order Functions

  //Below function calls possible use case is when we wanted to gather the debug info as we call series of functions
  def f0(a: Int): (Int, String) = {
    val result = a + 1
    (result, " f0")
  }

  def g0(a: Int): (Int, String) = {
    val result = a  + 2
    (result, " g0")
  }
  def h0(a: Int): (Int, String) = {
    val result = a + 3
    (result, " h0")
  } // bind, a HOF

  def bind(fun: (Int) => (Int, String),
           tup: Tuple2[Int, String]): (Int, String) =
    {
      val (intResult, stringResult) = fun(tup._1)
      (intResult, tup._2 + stringResult)
    }

  //Now lets see how to use for expression and make this in FP style

  def f(v: Int): DebuggableWrapper[Int] = {
    val res =  DebuggableWrapper(v + 1, "F")
    println("f(" + v +") -> " + res + "\n")
    res
  }
  def g(v: Int): DebuggableWrapper[Int] = {
    val res =  DebuggableWrapper(v + 1, "G")
    println("g(" + v +") -> " + res + "\n")
    res
  }
  def h(v: Int): DebuggableWrapper[Int] = {
    val res =  DebuggableWrapper(v + 1, "H")
    println("h(" + v +") -> " + res + "\n")
    res
  }

  def main(args: Array[String]): Unit = {

    val fResult = f0(0)
    val gResult = bind(g0, fResult)
    val hResult = bind(h0, gResult)

    println(s"result: ${hResult._1} debug: ${hResult._2}")

    println("-------------------------------------------------------")

    val res = for {
      i <- f(0)
      j <- g(i)
      k <- h(j)
    } yield k

    println(res)

    println("-----------------------------------------------------")

     //compiler generated : "scalac -Xprint:parse src/main/scala/io/dhiraa/Debuggable.scala "
    //val res = f(0).flatMap(((i) => g(i).flatMap(((j) => h(j).map(((k) => k))))));

    val res1 = f(0).flatMap { i =>
      println("This is called as part of lambda function with value of i as " + i)
      g(i).flatMap { j =>
        println("This is called as part of lambda function with value of j as " + j)
        h(j).map{k =>
          println("This is called as part of lambda function with value of k as " + k)
          k
        }
      }
    }

    println(res1)

    println("-----------------------------------------------------")

    val res3 = for {
      a <- State(20)
      b <- State(a + 15) //manually carry over `a`
      c <- State(b + 0) //manually carry over `b`
    } yield c
    println(s"res: $res3") //prints "State(35)"

    //val res3 = State(20).flatMap(((a) => State(a.$plus(15)).
    // flatMap(((b) => State(b.$plus(0)).map(((c) => c))))));

  }
}
