benchmark://user-v0/20250220T202319-e16e
; ModuleID = '-'
source_filename = "-"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @partition(i32* %0, i32 %1, i32 %2) #0 {
  %4 = alloca i32*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store i32* %0, i32** %4, align 8, !tbaa !0
  store i32 %1, i32* %5, align 4, !tbaa !4
  store i32 %2, i32* %6, align 4, !tbaa !4
  %12 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %12) #2
  %13 = load i32*, i32** %4, align 8, !tbaa !0
  %14 = load i32, i32* %6, align 4, !tbaa !4
  %15 = sext i32 %14 to i64
  %16 = getelementptr inbounds i32, i32* %13, i64 %15
  %17 = load i32, i32* %16, align 4, !tbaa !4
  store i32 %17, i32* %7, align 4, !tbaa !4
  %18 = bitcast i32* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %18) #2
  %19 = load i32, i32* %5, align 4, !tbaa !4
  %20 = sub nsw i32 %19, 1
  store i32 %20, i32* %8, align 4, !tbaa !4
  %21 = bitcast i32* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %21) #2
  %22 = load i32, i32* %5, align 4, !tbaa !4
  store i32 %22, i32* %9, align 4, !tbaa !4
  br label %23

23:                                               ; preds = %62, %3
  %24 = load i32, i32* %9, align 4, !tbaa !4
  %25 = load i32, i32* %6, align 4, !tbaa !4
  %26 = icmp slt i32 %24, %25
  br i1 %26, label %29, label %27

27:                                               ; preds = %23
  %28 = bitcast i32* %9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %28) #2
  br label %65

29:                                               ; preds = %23
  %30 = load i32*, i32** %4, align 8, !tbaa !0
  %31 = load i32, i32* %9, align 4, !tbaa !4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds i32, i32* %30, i64 %32
  %34 = load i32, i32* %33, align 4, !tbaa !4
  %35 = load i32, i32* %7, align 4, !tbaa !4
  %36 = icmp slt i32 %34, %35
  br i1 %36, label %37, label %61

37:                                               ; preds = %29
  %38 = load i32, i32* %8, align 4, !tbaa !4
  %39 = add nsw i32 %38, 1
  store i32 %39, i32* %8, align 4, !tbaa !4
  %40 = bitcast i32* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %40) #2
  %41 = load i32*, i32** %4, align 8, !tbaa !0
  %42 = load i32, i32* %8, align 4, !tbaa !4
  %43 = sext i32 %42 to i64
  %44 = getelementptr inbounds i32, i32* %41, i64 %43
  %45 = load i32, i32* %44, align 4, !tbaa !4
  store i32 %45, i32* %10, align 4, !tbaa !4
  %46 = load i32*, i32** %4, align 8, !tbaa !0
  %47 = load i32, i32* %9, align 4, !tbaa !4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds i32, i32* %46, i64 %48
  %50 = load i32, i32* %49, align 4, !tbaa !4
  %51 = load i32*, i32** %4, align 8, !tbaa !0
  %52 = load i32, i32* %8, align 4, !tbaa !4
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds i32, i32* %51, i64 %53
  store i32 %50, i32* %54, align 4, !tbaa !4
  %55 = load i32, i32* %10, align 4, !tbaa !4
  %56 = load i32*, i32** %4, align 8, !tbaa !0
  %57 = load i32, i32* %9, align 4, !tbaa !4
  %58 = sext i32 %57 to i64
  %59 = getelementptr inbounds i32, i32* %56, i64 %58
  store i32 %55, i32* %59, align 4, !tbaa !4
  %60 = bitcast i32* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %60) #2
  br label %61

61:                                               ; preds = %37, %29
  br label %62

62:                                               ; preds = %61
  %63 = load i32, i32* %9, align 4, !tbaa !4
  %64 = add nsw i32 %63, 1
  store i32 %64, i32* %9, align 4, !tbaa !4
  br label %23

65:                                               ; preds = %27
  %66 = bitcast i32* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %66) #2
  %67 = load i32*, i32** %4, align 8, !tbaa !0
  %68 = load i32, i32* %8, align 4, !tbaa !4
  %69 = add nsw i32 %68, 1
  %70 = sext i32 %69 to i64
  %71 = getelementptr inbounds i32, i32* %67, i64 %70
  %72 = load i32, i32* %71, align 4, !tbaa !4
  store i32 %72, i32* %11, align 4, !tbaa !4
  %73 = load i32*, i32** %4, align 8, !tbaa !0
  %74 = load i32, i32* %6, align 4, !tbaa !4
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds i32, i32* %73, i64 %75
  %77 = load i32, i32* %76, align 4, !tbaa !4
  %78 = load i32*, i32** %4, align 8, !tbaa !0
  %79 = load i32, i32* %8, align 4, !tbaa !4
  %80 = add nsw i32 %79, 1
  %81 = sext i32 %80 to i64
  %82 = getelementptr inbounds i32, i32* %78, i64 %81
  store i32 %77, i32* %82, align 4, !tbaa !4
  %83 = load i32, i32* %11, align 4, !tbaa !4
  %84 = load i32*, i32** %4, align 8, !tbaa !0
  %85 = load i32, i32* %6, align 4, !tbaa !4
  %86 = sext i32 %85 to i64
  %87 = getelementptr inbounds i32, i32* %84, i64 %86
  store i32 %83, i32* %87, align 4, !tbaa !4
  %88 = load i32, i32* %8, align 4, !tbaa !4
  %89 = add nsw i32 %88, 1
  %90 = bitcast i32* %11 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %90) #2
  %91 = bitcast i32* %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %91) #2
  %92 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %92) #2
  ret i32 %89
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local void @quickSort(i32* %0, i32 %1, i32 %2) #0 {
  %4 = alloca i32*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32* %0, i32** %4, align 8, !tbaa !0
  store i32 %1, i32* %5, align 4, !tbaa !4
  store i32 %2, i32* %6, align 4, !tbaa !4
  %8 = load i32, i32* %5, align 4, !tbaa !4
  %9 = load i32, i32* %6, align 4, !tbaa !4
  %10 = icmp slt i32 %8, %9
  br i1 %10, label %11, label %26

11:                                               ; preds = %3
  %12 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %12) #2
  %13 = load i32*, i32** %4, align 8, !tbaa !0
  %14 = load i32, i32* %5, align 4, !tbaa !4
  %15 = load i32, i32* %6, align 4, !tbaa !4
  %16 = call i32 @partition(i32* %13, i32 %14, i32 %15)
  store i32 %16, i32* %7, align 4, !tbaa !4
  %17 = load i32*, i32** %4, align 8, !tbaa !0
  %18 = load i32, i32* %5, align 4, !tbaa !4
  %19 = load i32, i32* %7, align 4, !tbaa !4
  %20 = sub nsw i32 %19, 1
  call void @quickSort(i32* %17, i32 %18, i32 %20)
  %21 = load i32*, i32** %4, align 8, !tbaa !0
  %22 = load i32, i32* %7, align 4, !tbaa !4
  %23 = add nsw i32 %22, 1
  %24 = load i32, i32* %6, align 4, !tbaa !4
  call void @quickSort(i32* %21, i32 %23, i32 %24)
  %25 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %25) #2
  br label %26

26:                                               ; preds = %11, %3
  ret void
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}